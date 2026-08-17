//! Model Generator
//!
//! Automatic generation of model architecture scaffolding and boilerplate code.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for model generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGeneratorConfig {
    /// Model name
    pub model_name: String,
    /// Model type (encoder, decoder, encoder-decoder)
    pub model_type: ModelType,
    /// Configuration parameters
    pub config_params: HashMap<String, ConfigParam>,
    /// Layer definitions
    pub layers: Vec<LayerDefinition>,
    /// Task heads to generate
    pub task_heads: Vec<TaskHead>,
}

/// Model architecture type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Encoder,
    Decoder,
    EncoderDecoder,
    Multimodal,
    Custom,
}

/// Configuration parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigParam {
    pub name: String,
    pub param_type: String,
    pub default_value: String,
    pub description: String,
}

/// Layer definition for model generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDefinition {
    pub name: String,
    pub layer_type: String,
    pub parameters: HashMap<String, String>,
}

/// Code fragments for one generated layer.
struct GeneratedLayer {
    /// Import path inside `trustformers_core::layers`.
    import: &'static str,
    /// Rust type of the struct field.
    field_type: &'static str,
    /// Expression that constructs the layer.
    constructor: String,
    /// Statement that applies the layer in the forward pass.
    forward: String,
}

/// Task head definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskHead {
    pub name: String,
    pub task_type: String,
    pub output_size: Option<usize>,
}

/// Model generator
pub struct ModelGenerator {
    config: ModelGeneratorConfig,
}

impl ModelGenerator {
    /// Create a new model generator
    pub fn new(config: ModelGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate model architecture code
    pub fn generate_model(&self, output_dir: &Path) -> Result<()> {
        // Create output directory structure
        std::fs::create_dir_all(output_dir)?;

        let model_dir = output_dir.join(&self.config.model_name);
        std::fs::create_dir_all(&model_dir)?;

        // Generate configuration file
        self.generate_config_file(&model_dir)?;

        // Generate model implementation
        self.generate_model_file(&model_dir)?;

        // Generate module file
        self.generate_mod_file(&model_dir)?;

        // Generate test file
        self.generate_test_file(&model_dir)?;

        Ok(())
    }

    /// Generate configuration file
    fn generate_config_file(&self, output_dir: &Path) -> Result<()> {
        let config_content = self.generate_config_code();
        let config_path = output_dir.join("config.rs");
        std::fs::write(config_path, config_content)?;
        Ok(())
    }

    /// Generate model implementation file
    fn generate_model_file(&self, output_dir: &Path) -> Result<()> {
        let model_content = self.generate_model_code()?;
        let model_path = output_dir.join("model.rs");
        std::fs::write(model_path, model_content)?;
        Ok(())
    }

    /// Generate module file
    fn generate_mod_file(&self, output_dir: &Path) -> Result<()> {
        let mod_content = format!(
            "//! {} Model Implementation\n\npub mod config;\npub mod model;\n\npub use config::{}Config;\npub use model::{}Model;\n",
            self.config.model_name,
            self.config.model_name,
            self.config.model_name
        );
        let mod_path = output_dir.join("mod.rs");
        std::fs::write(mod_path, mod_content)?;
        Ok(())
    }

    /// Generate test file
    fn generate_test_file(&self, output_dir: &Path) -> Result<()> {
        let test_content = self.generate_test_code();
        let test_path = output_dir.join("tests.rs");
        std::fs::write(test_path, test_content)?;
        Ok(())
    }

    /// Generate configuration code
    fn generate_config_code(&self) -> String {
        let mut code = format!(
            "//! {} Configuration\n\nuse serde::{{Deserialize, Serialize}};\n\n#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct {}Config {{\n",
            self.config.model_name,
            self.config.model_name
        );

        // Add configuration parameters
        for param in self.config.config_params.values() {
            code.push_str(&format!(
                "    /// {}\n    pub {}: {},\n",
                param.description, param.name, param.param_type
            ));
        }

        code.push_str("}\n\n");

        // Add Default implementation
        code.push_str(&format!(
            "impl Default for {}Config {{\n    fn default() -> Self {{\n        Self {{\n",
            self.config.model_name
        ));

        for param in self.config.config_params.values() {
            code.push_str(&format!(
                "            {}: {},\n",
                param.name, param.default_value
            ));
        }

        code.push_str("        }\n    }\n}\n");

        code
    }

    /// Description of a layer type this generator can emit.
    ///
    /// Every entry maps onto a type that really exists in
    /// [`trustformers_core::layers`]; there is no entry for layers core does not
    /// provide, so the generator can never emit an import that will not resolve.
    fn layer_spec(layer: &LayerDefinition) -> Result<Option<GeneratedLayer>> {
        let parameter = |key: &str, fallback: &str| -> String {
            layer.parameters.get(key).cloned().unwrap_or_else(|| fallback.to_string())
        };

        let spec = match layer.layer_type.as_str() {
            "linear" => GeneratedLayer {
                import: "linear::Linear",
                field_type: "Linear",
                // `Linear::new` is infallible.
                constructor: format!(
                    "Linear::new({}, {}, {})",
                    parameter("input_size", "768"),
                    parameter("output_size", "768"),
                    parameter("bias", "true")
                ),
                forward: format!("let x = self.{}.forward(x)?;", layer.name),
            },
            "attention" => GeneratedLayer {
                import: "attention::MultiHeadAttention",
                field_type: "MultiHeadAttention",
                constructor: format!(
                    "MultiHeadAttention::new({}, {}, {}, {})?",
                    parameter("hidden_size", "768"),
                    parameter("num_heads", "12"),
                    parameter("dropout_prob", "0.1"),
                    parameter("bias", "true")
                ),
                forward: format!(
                    "let x = self.{}.forward_self_attention(&x, None, false)?;",
                    layer.name
                ),
            },
            "conv2d" => GeneratedLayer {
                import: "conv2d::Conv2d",
                field_type: "Conv2d",
                constructor: format!(
                    "Conv2d::new({}, {}, ({k}, {k}), ({s}, {s}), ({p}, {p}), {bias})?",
                    parameter("in_channels", "1"),
                    parameter("out_channels", "64"),
                    k = parameter("kernel_size", "3"),
                    s = parameter("stride", "1"),
                    p = parameter("padding", "0"),
                    bias = parameter("bias", "true")
                ),
                forward: format!("let x = self.{}.forward(x)?;", layer.name),
            },
            "layernorm" => GeneratedLayer {
                import: "layernorm::LayerNorm",
                field_type: "LayerNorm",
                constructor: format!(
                    "LayerNorm::new(vec![{}], {})?",
                    parameter("normalized_shape", "768"),
                    parameter("eps", "1e-5")
                ),
                forward: format!("let x = self.{}.forward(x)?;", layer.name),
            },
            "rmsnorm" => GeneratedLayer {
                import: "layernorm::RMSNorm",
                field_type: "RMSNorm",
                constructor: format!(
                    "RMSNorm::new({}, {})?",
                    parameter("hidden_size", "768"),
                    parameter("eps", "1e-6")
                ),
                forward: format!("let x = self.{}.forward(x)?;", layer.name),
            },
            "dropout" => GeneratedLayer {
                import: "dropout::Dropout",
                field_type: "Dropout",
                // `Dropout::new` is infallible.
                constructor: format!("Dropout::new({})", parameter("p", "0.1")),
                forward: format!("let x = self.{}.forward(x)?;", layer.name),
            },
            "embedding" => GeneratedLayer {
                import: "embedding::Embedding",
                field_type: "Embedding",
                constructor: format!(
                    "Embedding::new({}, {}, {})?",
                    parameter("num_embeddings", "30522"),
                    parameter("embedding_dim", "768"),
                    parameter("padding_idx", "None")
                ),
                // `Embedding` consumes token ids, not a tensor, so it is applied
                // to the raw input before the tensor pipeline starts.
                forward: format!("let x = self.{}.forward(input_ids.to_vec())?;", layer.name),
            },
            "feedforward" => GeneratedLayer {
                import: "feedforward::FeedForward",
                field_type: "FeedForward",
                constructor: format!(
                    "FeedForward::new({}, {}, {})?",
                    parameter("hidden_size", "768"),
                    parameter("intermediate_size", "3072"),
                    parameter("dropout_prob", "0.1")
                ),
                forward: format!("let x = self.{}.forward(x)?;", layer.name),
            },
            // Activations are tensor methods: no field, no import.
            "relu" => return Ok(None),
            "gelu" => return Ok(None),
            "silu" => return Ok(None),
            "tanh" => return Ok(None),
            "sigmoid" => return Ok(None),
            "softmax" => return Ok(None),
            other => {
                return Err(anyhow!(
                    "layer type `{other}` (layer `{}`) has no implementation in \
                     trustformers_core::layers, so no compilable code can be generated for it. \
                     Supported types: linear, attention, conv2d, layernorm, rmsnorm, dropout, \
                     embedding, feedforward, relu, gelu, silu, tanh, sigmoid, softmax",
                    layer.name
                ))
            },
        };

        Ok(Some(spec))
    }

    /// Forward-pass expression for an activation layer type.
    fn activation_call(layer_type: &str) -> Option<&'static str> {
        match layer_type {
            "relu" => Some("let x = x.relu()?;"),
            "gelu" => Some("let x = x.gelu()?;"),
            "silu" => Some("let x = x.silu()?;"),
            "tanh" => Some("let x = x.tanh()?;"),
            "sigmoid" => Some("let x = x.sigmoid()?;"),
            "softmax" => Some("let x = x.softmax(1)?;"),
            _ => None,
        }
    }

    /// Generate model code
    ///
    /// # Errors
    ///
    /// Fails when the configuration requests a layer type that has no
    /// implementation in `trustformers_core::layers`: emitting an import for a
    /// module that does not exist would produce a file that cannot compile.
    fn generate_model_code(&self) -> Result<String> {
        let forward_impl = self.generate_forward_implementation()?;
        let (layer_fields, layer_init, imports) = self.generate_layers_code()?;

        let import_block = if imports.is_empty() {
            String::new()
        } else {
            format!(
                "use trustformers_core::layers::{{{}}};\nuse trustformers_core::traits::Layer;\n",
                imports.join(", ")
            )
        };

        Ok(format!(
            "//! {name} Model Implementation\n\
             //!\n\
             //! This file is auto-generated by `trustformers_models::developer_tools`.\n\n\
             use super::config::{name}Config;\n\
             use trustformers_core::errors::Result;\n\
             use trustformers_core::tensor::Tensor;\n\
             {imports}\n\
             #[derive(Debug, Clone)]\n\
             pub struct {name}Model {{\n    config: {name}Config,{fields}\n}}\n\n\
             impl {name}Model {{\n\
             \x20   pub fn new(config: {name}Config) -> Result<Self> {{\n\
             \x20       Ok(Self {{\n            config,{init}\n        }})\n    }}\n\n\
             \x20   pub fn config(&self) -> &{name}Config {{\n        &self.config\n    }}\n\n\
             \x20   pub fn forward(&self, input: &Tensor) -> Result<Tensor> {{\n{forward}\n    }}\n\
             }}\n",
            name = self.config.model_name,
            imports = import_block,
            fields = layer_fields,
            init = layer_init,
            forward = forward_impl
        ))
    }

    /// Generate forward pass implementation based on model type and layers
    fn generate_forward_implementation(&self) -> Result<String> {
        match self.config.model_type {
            ModelType::Encoder => self.generate_encoder_forward(),
            ModelType::Decoder => Ok(self.generate_decoder_forward()),
            ModelType::EncoderDecoder => Ok(self.generate_encoder_decoder_forward()),
            ModelType::Multimodal => Ok(self.generate_multimodal_forward()),
            ModelType::Custom => self.generate_custom_forward(),
        }
    }

    /// Generate encoder forward pass
    fn generate_encoder_forward(&self) -> Result<String> {
        let mut layer_calls = Vec::with_capacity(self.config.layers.len());
        for layer in &self.config.layers {
            if let Some(activation) = Self::activation_call(&layer.layer_type) {
                layer_calls.push(format!("        {activation}"));
                continue;
            }
            let spec = Self::layer_spec(layer)?
                .ok_or_else(|| anyhow!("layer `{}` produced no code to call", layer.name))?;
            layer_calls.push(format!("        {}", spec.forward));
        }

        Ok(format!(
            "        let x = input.clone();\n{}\n\n        Ok(x)",
            layer_calls.join("\n")
        ))
    }

    /// Generate decoder forward pass
    fn generate_decoder_forward(&self) -> String {
        "        let mut x = input.clone();\n        \n        // Decoder layers with causal masking\n        for i in 0..self.config.num_layers {\n            // Self-attention with causal mask\n            // Feed-forward network\n        }\n        \n        Ok(x)".to_string()
    }

    /// Generate encoder-decoder forward pass
    fn generate_encoder_decoder_forward(&self) -> String {
        "        let mut encoder_output = input.clone();\n        \n        // Encoder pass\n        for i in 0..self.config.encoder_layers {\n            // Encoder self-attention and FFN\n        }\n        \n        // Decoder pass with cross-attention\n        let mut decoder_output = encoder_output.clone();\n        for i in 0..self.config.decoder_layers {\n            // Decoder self-attention, cross-attention, and FFN\n        }\n        \n        Ok(decoder_output)".to_string()
    }

    /// Generate multimodal forward pass
    fn generate_multimodal_forward(&self) -> String {
        "        // Extract different modalities from input\n        let text_features = input.slice(1, 0, self.config.text_dim)?;\n        let visual_features = input.slice(1, self.config.text_dim, input.shape()[1])?;\n        \n        // Process each modality\n        let text_output = self.process_text_modality(&text_features)?;\n        let visual_output = self.process_visual_modality(&visual_features)?;\n        \n        // Fusion layer\n        let fused = Tensor::concat(&[text_output, visual_output], 1)?;\n        \n        Ok(fused)".to_string()
    }

    /// Generate custom forward pass
    fn generate_custom_forward(&self) -> Result<String> {
        if self.config.layers.is_empty() {
            Ok("        // Custom model implementation: no layers were configured,\n        // so the generated forward pass is the identity.\n        let output = input.clone();\n\n        Ok(output)".to_string())
        } else {
            self.generate_encoder_forward() // Default to encoder-style for custom with layers
        }
    }

    /// Generate layer field declarations, initialisation code and the exact set
    /// of imports the generated file needs.
    ///
    /// Only layer types that exist in `trustformers_core::layers` are emitted;
    /// anything else is an error rather than an import that cannot resolve.
    fn generate_layers_code(&self) -> Result<(String, String, Vec<String>)> {
        if self.config.layers.is_empty() {
            return Ok((String::new(), String::new(), Vec::new()));
        }

        let mut fields = String::new();
        let mut init = String::new();
        let mut imports: Vec<String> = Vec::new();

        for layer in &self.config.layers {
            let Some(spec) = Self::layer_spec(layer)? else {
                // Activation: no field, no import.
                continue;
            };

            fields.push_str(&format!("\n    {}: {},", layer.name, spec.field_type));
            init.push_str(&format!(
                "\n            {}: {},",
                layer.name, spec.constructor
            ));

            let import = spec.import.to_string();
            if !imports.contains(&import) {
                imports.push(import);
            }
        }

        imports.sort();
        Ok((fields, init, imports))
    }

    /// Generate test code
    fn generate_test_code(&self) -> String {
        format!(
            "//! {} Tests\n\nuse super::{{{}Config, {}Model}};\n\n#[test]\nfn test_{}_creation() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"operation failed\");\n    // Add assertions here\n}}\n",
            self.config.model_name,
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name
        )
    }
}

/// Predefined model templates
pub struct ModelTemplates;

impl ModelTemplates {
    /// Get BERT-style encoder template
    pub fn bert_encoder() -> ModelGeneratorConfig {
        let mut config_params = HashMap::new();
        config_params.insert(
            "vocab_size".to_string(),
            ConfigParam {
                name: "vocab_size".to_string(),
                param_type: "usize".to_string(),
                default_value: "30522".to_string(),
                description: "Vocabulary size".to_string(),
            },
        );
        config_params.insert(
            "hidden_size".to_string(),
            ConfigParam {
                name: "hidden_size".to_string(),
                param_type: "usize".to_string(),
                default_value: "768".to_string(),
                description: "Hidden dimension size".to_string(),
            },
        );

        ModelGeneratorConfig {
            model_name: "CustomBert".to_string(),
            model_type: ModelType::Encoder,
            config_params,
            layers: vec![],
            task_heads: vec![],
        }
    }

    /// Get GPT-style decoder template
    pub fn gpt_decoder() -> ModelGeneratorConfig {
        let mut config_params = HashMap::new();
        config_params.insert(
            "vocab_size".to_string(),
            ConfigParam {
                name: "vocab_size".to_string(),
                param_type: "usize".to_string(),
                default_value: "50257".to_string(),
                description: "Vocabulary size".to_string(),
            },
        );

        ModelGeneratorConfig {
            model_name: "CustomGPT".to_string(),
            model_type: ModelType::Decoder,
            config_params,
            layers: vec![],
            task_heads: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_generator_config_creation() {
        let config = ModelGeneratorConfig {
            model_name: "TestModel".to_string(),
            model_type: ModelType::Encoder,
            config_params: HashMap::new(),
            layers: vec![],
            task_heads: vec![],
        };
        assert_eq!(config.model_name, "TestModel");
        assert!(config.config_params.is_empty());
        assert!(config.layers.is_empty());
        assert!(config.task_heads.is_empty());
    }

    #[test]
    fn test_model_type_variants() {
        let types = vec![
            ModelType::Encoder,
            ModelType::Decoder,
            ModelType::EncoderDecoder,
            ModelType::Multimodal,
            ModelType::Custom,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn test_config_param_creation() {
        let param = ConfigParam {
            name: "hidden_size".to_string(),
            param_type: "usize".to_string(),
            default_value: "768".to_string(),
            description: "Hidden dimension size".to_string(),
        };
        assert_eq!(param.name, "hidden_size");
        assert_eq!(param.param_type, "usize");
        assert_eq!(param.default_value, "768");
    }

    #[test]
    fn test_layer_definition_creation() {
        let layer = LayerDefinition {
            name: "self_attention".to_string(),
            layer_type: "attention".to_string(),
            parameters: HashMap::from([
                ("num_heads".to_string(), "12".to_string()),
                ("hidden_size".to_string(), "768".to_string()),
            ]),
        };
        assert_eq!(layer.name, "self_attention");
        assert_eq!(layer.layer_type, "attention");
        assert_eq!(layer.parameters.len(), 2);
    }

    #[test]
    fn test_task_head_creation() {
        let head = TaskHead {
            name: "classification".to_string(),
            task_type: "sequence_classification".to_string(),
            output_size: Some(10),
        };
        assert_eq!(head.name, "classification");
        assert_eq!(head.output_size, Some(10));
    }

    #[test]
    fn test_task_head_no_output_size() {
        let head = TaskHead {
            name: "lm_head".to_string(),
            task_type: "language_modeling".to_string(),
            output_size: None,
        };
        assert!(head.output_size.is_none());
    }

    #[test]
    fn test_model_generator_new() {
        let config = ModelGeneratorConfig {
            model_name: "MyModel".to_string(),
            model_type: ModelType::Decoder,
            config_params: HashMap::new(),
            layers: vec![],
            task_heads: vec![],
        };
        let _generator = ModelGenerator::new(config);
    }

    #[test]
    fn test_bert_encoder_template() {
        let config = ModelTemplates::bert_encoder();
        assert_eq!(config.model_name, "CustomBert");
        assert!(matches!(config.model_type, ModelType::Encoder));
        assert!(config.config_params.contains_key("vocab_size"));
        assert!(config.config_params.contains_key("hidden_size"));
    }

    #[test]
    fn test_gpt_decoder_template() {
        let config = ModelTemplates::gpt_decoder();
        assert_eq!(config.model_name, "CustomGPT");
        assert!(matches!(config.model_type, ModelType::Decoder));
        assert!(config.config_params.contains_key("vocab_size"));
    }

    #[test]
    fn test_bert_template_defaults() {
        let config = ModelTemplates::bert_encoder();
        let vocab_param = config.config_params.get("vocab_size").expect("vocab_size not found");
        assert_eq!(vocab_param.default_value, "30522");
        let hidden_param = config.config_params.get("hidden_size").expect("hidden_size not found");
        assert_eq!(hidden_param.default_value, "768");
    }

    #[test]
    fn test_gpt_template_defaults() {
        let config = ModelTemplates::gpt_decoder();
        let vocab_param = config.config_params.get("vocab_size").expect("vocab_size not found");
        assert_eq!(vocab_param.default_value, "50257");
    }

    #[test]
    fn test_model_generator_generate_to_temp_dir() {
        let config = ModelTemplates::bert_encoder();
        let generator = ModelGenerator::new(config);
        let temp_dir = std::env::temp_dir().join("test_model_gen");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let result = generator.generate_model(&temp_dir);
        assert!(result.is_ok());
        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_model_generator_creates_files() {
        let config = ModelTemplates::gpt_decoder();
        let generator = ModelGenerator::new(config);
        let temp_dir = std::env::temp_dir().join("test_model_gen_files");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let result = generator.generate_model(&temp_dir);
        assert!(result.is_ok());
        // Check files were created
        let model_dir = temp_dir.join("CustomGPT");
        assert!(model_dir.exists());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_config_with_layers() {
        let config = ModelGeneratorConfig {
            model_name: "LayeredModel".to_string(),
            model_type: ModelType::Encoder,
            config_params: HashMap::new(),
            layers: vec![
                LayerDefinition {
                    name: "embedding".to_string(),
                    layer_type: "embedding".to_string(),
                    parameters: HashMap::from([
                        ("vocab_size".to_string(), "30000".to_string()),
                        ("hidden_size".to_string(), "512".to_string()),
                    ]),
                },
                LayerDefinition {
                    name: "encoder".to_string(),
                    layer_type: "attention".to_string(),
                    parameters: HashMap::from([
                        ("num_heads".to_string(), "8".to_string()),
                        ("hidden_size".to_string(), "512".to_string()),
                    ]),
                },
            ],
            task_heads: vec![],
        };
        assert_eq!(config.layers.len(), 2);
        assert_eq!(config.layers[0].name, "embedding");
        assert_eq!(config.layers[1].name, "encoder");
    }

    #[test]
    fn test_config_with_task_heads() {
        let config = ModelGeneratorConfig {
            model_name: "MultiTask".to_string(),
            model_type: ModelType::Encoder,
            config_params: HashMap::new(),
            layers: vec![],
            task_heads: vec![
                TaskHead {
                    name: "cls".to_string(),
                    task_type: "classification".to_string(),
                    output_size: Some(10),
                },
                TaskHead {
                    name: "ner".to_string(),
                    task_type: "token_classification".to_string(),
                    output_size: Some(9),
                },
            ],
        };
        assert_eq!(config.task_heads.len(), 2);
    }

    #[test]
    fn test_generator_with_attention_layer() {
        let mut params = HashMap::new();
        params.insert("num_heads".to_string(), "8".to_string());
        params.insert("hidden_size".to_string(), "512".to_string());
        let layer = LayerDefinition {
            name: "self_attn".to_string(),
            layer_type: "attention".to_string(),
            parameters: params,
        };
        assert_eq!(layer.parameters["num_heads"], "8");
    }

    #[test]
    fn test_generator_with_feedforward_layer() {
        let mut params = HashMap::new();
        params.insert("input_size".to_string(), "512".to_string());
        params.insert("hidden_size".to_string(), "2048".to_string());
        let layer = LayerDefinition {
            name: "ffn".to_string(),
            layer_type: "feedforward".to_string(),
            parameters: params,
        };
        assert_eq!(layer.parameters["hidden_size"], "2048");
    }

    #[test]
    fn test_generator_with_linear_layer() {
        let layer = LayerDefinition {
            name: "linear".to_string(),
            layer_type: "linear".to_string(),
            parameters: HashMap::from([
                ("input_size".to_string(), "768".to_string()),
                ("output_size".to_string(), "3072".to_string()),
            ]),
        };
        assert_eq!(layer.layer_type, "linear");
    }

    #[test]
    fn test_generator_with_conv1d_layer() {
        let layer = LayerDefinition {
            name: "conv".to_string(),
            layer_type: "conv1d".to_string(),
            parameters: HashMap::from([
                ("in_channels".to_string(), "768".to_string()),
                ("out_channels".to_string(), "768".to_string()),
            ]),
        };
        assert_eq!(layer.layer_type, "conv1d");
    }

    #[test]
    fn test_config_param_types() {
        let params = vec![
            ConfigParam {
                name: "int_param".to_string(),
                param_type: "usize".to_string(),
                default_value: "42".to_string(),
                description: "An integer".to_string(),
            },
            ConfigParam {
                name: "float_param".to_string(),
                param_type: "f32".to_string(),
                default_value: "0.1".to_string(),
                description: "A float".to_string(),
            },
            ConfigParam {
                name: "bool_param".to_string(),
                param_type: "bool".to_string(),
                default_value: "true".to_string(),
                description: "A boolean".to_string(),
            },
            ConfigParam {
                name: "str_param".to_string(),
                param_type: "String".to_string(),
                default_value: "hello".to_string(),
                description: "A string".to_string(),
            },
        ];
        assert_eq!(params.len(), 4);
        for p in &params {
            assert!(!p.name.is_empty());
            assert!(!p.param_type.is_empty());
        }
    }

    #[test]
    fn test_generate_test_code() {
        let config = ModelTemplates::bert_encoder();
        let generator = ModelGenerator::new(config);
        let test_code = generator.generate_test_code();
        assert!(test_code.contains("test_"));
        assert!(test_code.contains("CustomBert"));
    }

    #[test]
    fn test_model_name_in_generated_output() {
        let config = ModelGeneratorConfig {
            model_name: "UniqueTestModel".to_string(),
            model_type: ModelType::Encoder,
            config_params: HashMap::new(),
            layers: vec![],
            task_heads: vec![],
        };
        let generator = ModelGenerator::new(config);
        let test_code = generator.generate_test_code();
        assert!(test_code.contains("UniqueTestModel"));
    }

    // ------------------------------------------------------------------
    // Regression tests: generated code must reference real core modules
    // ------------------------------------------------------------------

    fn layer(name: &str, layer_type: &str) -> LayerDefinition {
        LayerDefinition {
            name: name.to_string(),
            layer_type: layer_type.to_string(),
            parameters: HashMap::new(),
        }
    }

    fn generator_with_layers(layers: Vec<LayerDefinition>) -> ModelGenerator {
        ModelGenerator::new(ModelGeneratorConfig {
            model_name: "Generated".to_string(),
            model_type: ModelType::Encoder,
            config_params: HashMap::new(),
            layers,
            task_heads: vec![],
        })
    }

    #[test]
    fn test_generated_imports_reference_existing_core_modules() {
        let generator = generator_with_layers(vec![
            layer("embed", "embedding"),
            layer("norm", "layernorm"),
            layer("attn", "attention"),
            layer("proj", "linear"),
            layer("drop", "dropout"),
        ]);

        let code = generator.generate_model_code().expect("model code generation");

        // Modules that exist in trustformers_core::layers.
        for expected in [
            "embedding::Embedding",
            "layernorm::LayerNorm",
            "attention::MultiHeadAttention",
            "linear::Linear",
            "dropout::Dropout",
        ] {
            assert!(
                code.contains(expected),
                "generated code must import {expected}:\n{code}"
            );
        }

        // Modules that do NOT exist and must never be emitted.
        for forbidden in [
            "layers::conv::",
            "conv::{",
            "normalization::",
            "transformer::TransformerBlock",
            "rnn::",
            "PositionalEncoding",
            "BatchNorm",
            "Conv1d",
            "LSTM",
            "GRU",
        ] {
            assert!(
                !code.contains(forbidden),
                "generated code must not reference `{forbidden}`, which does not exist in \
                 trustformers_core::layers:\n{code}"
            );
        }
    }

    #[test]
    fn test_generated_imports_are_limited_to_the_configured_layers() {
        let generator = generator_with_layers(vec![layer("proj", "linear")]);
        let code = generator.generate_model_code().expect("model code generation");

        assert!(code.contains("linear::Linear"));
        for unused in [
            "Dropout",
            "MultiHeadAttention",
            "Conv2d",
            "Embedding",
            "FeedForward",
            "RMSNorm",
        ] {
            assert!(
                !code.contains(unused),
                "an unused import for `{unused}` would make the generated file warn:\n{code}"
            );
        }
    }

    #[test]
    fn test_activation_layers_need_no_field_or_import() {
        let generator = generator_with_layers(vec![layer("act", "gelu")]);
        let code = generator.generate_model_code().expect("model code generation");

        assert!(code.contains("x.gelu()?"), "{code}");
        assert!(
            !code.contains("use trustformers_core::layers::"),
            "an activation needs no layer import:\n{code}"
        );
    }

    #[test]
    fn test_unsupported_layer_type_is_rejected() {
        for unsupported in [
            "lstm",
            "gru",
            "rnn",
            "conv1d",
            "batchnorm",
            "transformer_block",
        ] {
            let generator = generator_with_layers(vec![layer("x", unsupported)]);
            let error = generator
                .generate_model_code()
                .expect_err("generating code for a non-existent layer must fail");
            assert!(
                error.to_string().contains(unsupported),
                "the error must name the unsupported layer type: {error}"
            );
        }
    }

    #[test]
    fn test_generate_model_propagates_unsupported_layers() {
        let generator = generator_with_layers(vec![layer("recurrent", "lstm")]);
        let temp_dir = std::env::temp_dir().join("trustformers_model_generator_unsupported");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let result = generator.generate_model(&temp_dir);
        assert!(
            result.is_err(),
            "a model that cannot be generated must not be written to disk"
        );
        assert!(
            !temp_dir.join("Generated").join("model.rs").exists(),
            "no model.rs may be written for an unsupported configuration"
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
