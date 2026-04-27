//! Model Generator
//!
//! Automatic generation of model architecture scaffolding and boilerplate code.

use anyhow::Result;
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
        let model_content = self.generate_model_code();
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

    /// Generate model code
    fn generate_model_code(&self) -> String {
        let forward_impl = self.generate_forward_implementation();
        let layers_code = self.generate_layers_code();

        format!(
            "//! {} Model Implementation\n\nuse super::config::{}Config;\nuse trustformers_core::errors::Result;\nuse trustformers_core::tensor::Tensor;\nuse trustformers_core::layers::{{\n    linear::Linear,\n    attention::MultiHeadAttention,\n    conv::{{Conv1d, Conv2d}},\n    normalization::{{BatchNorm, LayerNorm}},\n    dropout::Dropout,\n    embedding::{{Embedding, PositionalEncoding}},\n    transformer::TransformerBlock,\n    rnn::{{RNN, LSTM, GRU}},\n}};\n\n#[derive(Debug, Clone)]\npub struct {}Model {{\n    config: {}Config,{}\n}}\n\nimpl {}Model {{\n    pub fn new(config: {}Config) -> Result<Self> {{\n        Ok(Self {{\n            config,{}\n        }})\n    }}\n    \n    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {{\n{}\n    }}\n}}\n",
            self.config.model_name,
            self.config.model_name,
            self.config.model_name,
            self.config.model_name,
            layers_code.0, // layer fields
            self.config.model_name,
            self.config.model_name,
            layers_code.1, // layer initialization
            forward_impl
        )
    }

    /// Generate forward pass implementation based on model type and layers
    fn generate_forward_implementation(&self) -> String {
        match self.config.model_type {
            ModelType::Encoder => self.generate_encoder_forward(),
            ModelType::Decoder => self.generate_decoder_forward(),
            ModelType::EncoderDecoder => self.generate_encoder_decoder_forward(),
            ModelType::Multimodal => self.generate_multimodal_forward(),
            ModelType::Custom => self.generate_custom_forward(),
        }
    }

    /// Generate encoder forward pass
    fn generate_encoder_forward(&self) -> String {
        let layer_calls = self
            .config
            .layers
            .iter()
            .map(|layer| match layer.layer_type.as_str() {
                "linear" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "attention" => format!(
                    "        let x = self.{}.forward(&x, None, None)?;",
                    layer.name
                ),
                "layernorm" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "dropout" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "conv1d" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "conv2d" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "batchnorm" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "relu" => "        let x = x.relu()?;".to_string(),
                "gelu" => "        let x = x.gelu()?;".to_string(),
                "silu" => "        let x = x.silu()?;".to_string(),
                "tanh" => "        let x = x.tanh()?;".to_string(),
                "sigmoid" => "        let x = x.sigmoid()?;".to_string(),
                "softmax" => "        let x = x.softmax(1)?;".to_string(),
                "embedding" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "positional_encoding" => {
                    format!("        let x = self.{}.forward(&x)?;", layer.name)
                },
                "transformer_block" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "rnn" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "lstm" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                "gru" => format!("        let x = self.{}.forward(&x)?;", layer.name),
                _ => format!(
                    "        // Unsupported layer type '{}' - please implement manually",
                    layer.layer_type
                ),
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "        let mut x = input.clone();\n{}\n        \n        // Apply task heads if configured\n        for head in &self.config.task_heads {{\n            // Apply task-specific transformations\n        }}\n        \n        Ok(x)"
        , layer_calls)
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
    fn generate_custom_forward(&self) -> String {
        if self.config.layers.is_empty() {
            "        // Custom model implementation\n        // Please implement the forward pass based on your specific requirements\n        let output = input.clone();\n        \n        Ok(output)".to_string()
        } else {
            self.generate_encoder_forward() // Default to encoder-style for custom with layers
        }
    }

    /// Generate layer definitions and initialization code
    fn generate_layers_code(&self) -> (String, String) {
        if self.config.layers.is_empty() {
            return (String::new(), String::new());
        }

        let layer_fields = self
            .config
            .layers
            .iter()
            .map(|layer| match layer.layer_type.as_str() {
                "linear" => format!("\n    {}: Linear,", layer.name),
                "attention" => format!("\n    {}: MultiHeadAttention,", layer.name),
                "conv1d" => format!("\n    {}: Conv1d,", layer.name),
                "conv2d" => format!("\n    {}: Conv2d,", layer.name),
                "batchnorm" => format!("\n    {}: BatchNorm,", layer.name),
                "layernorm" => format!("\n    {}: LayerNorm,", layer.name),
                "dropout" => format!("\n    {}: Dropout,", layer.name),
                "embedding" => format!("\n    {}: Embedding,", layer.name),
                "positional_encoding" => format!("\n    {}: PositionalEncoding,", layer.name),
                "transformer_block" => format!("\n    {}: TransformerBlock,", layer.name),
                "rnn" => format!("\n    {}: RNN,", layer.name),
                "lstm" => format!("\n    {}: LSTM,", layer.name),
                "gru" => format!("\n    {}: GRU,", layer.name),
                "relu" | "gelu" | "silu" | "tanh" | "sigmoid" | "softmax" => String::new(), // Activation functions don't need fields
                _ => format!(
                    "\n    // Unsupported: {}: {},",
                    layer.name, layer.layer_type
                ),
            })
            .collect::<Vec<_>>()
            .join("");

        let layer_init = self
            .config
            .layers
            .iter()
            .map(|layer| match layer.layer_type.as_str() {
                "linear" => {
                    let default_768 = "768".to_string();
                    let input_size = layer.parameters.get("input_size").unwrap_or(&default_768);
                    let output_size = layer.parameters.get("output_size").unwrap_or(&default_768);
                    format!(
                        "\n            {}: Linear::new({}, {})?,",
                        layer.name, input_size, output_size
                    )
                },
                "attention" => {
                    let default_768 = "768".to_string();
                    let default_12 = "12".to_string();
                    let hidden_size = layer.parameters.get("hidden_size").unwrap_or(&default_768);
                    let num_heads = layer.parameters.get("num_heads").unwrap_or(&default_12);
                    format!(
                        "\n            {}: MultiHeadAttention::new({}, {})?,",
                        layer.name, hidden_size, num_heads
                    )
                },
                "conv1d" => {
                    let default_1 = "1".to_string();
                    let default_64 = "64".to_string();
                    let default_3 = "3".to_string();
                    let in_channels = layer.parameters.get("in_channels").unwrap_or(&default_1);
                    let out_channels = layer.parameters.get("out_channels").unwrap_or(&default_64);
                    let kernel_size = layer.parameters.get("kernel_size").unwrap_or(&default_3);
                    format!(
                        "\n            {}: Conv1d::new({}, {}, {})?,",
                        layer.name, in_channels, out_channels, kernel_size
                    )
                },
                "conv2d" => {
                    let default_1 = "1".to_string();
                    let default_64 = "64".to_string();
                    let default_3 = "3".to_string();
                    let in_channels = layer.parameters.get("in_channels").unwrap_or(&default_1);
                    let out_channels = layer.parameters.get("out_channels").unwrap_or(&default_64);
                    let kernel_size = layer.parameters.get("kernel_size").unwrap_or(&default_3);
                    format!(
                        "\n            {}: Conv2d::new({}, {}, {})?,",
                        layer.name, in_channels, out_channels, kernel_size
                    )
                },
                "batchnorm" => {
                    let default_768 = "768".to_string();
                    let num_features = layer.parameters.get("num_features").unwrap_or(&default_768);
                    format!(
                        "\n            {}: BatchNorm::new({})?,",
                        layer.name, num_features
                    )
                },
                "layernorm" => {
                    let default_768 = "768".to_string();
                    let normalized_shape =
                        layer.parameters.get("normalized_shape").unwrap_or(&default_768);
                    format!(
                        "\n            {}: LayerNorm::new({})?,",
                        layer.name, normalized_shape
                    )
                },
                "dropout" => {
                    let default_01 = "0.1".to_string();
                    let p = layer.parameters.get("p").unwrap_or(&default_01);
                    format!("\n            {}: Dropout::new({})?,", layer.name, p)
                },
                "embedding" => {
                    let default_30522 = "30522".to_string();
                    let default_768 = "768".to_string();
                    let num_embeddings =
                        layer.parameters.get("num_embeddings").unwrap_or(&default_30522);
                    let embedding_dim =
                        layer.parameters.get("embedding_dim").unwrap_or(&default_768);
                    format!(
                        "\n            {}: Embedding::new({}, {})?,",
                        layer.name, num_embeddings, embedding_dim
                    )
                },
                "positional_encoding" => {
                    let default_768 = "768".to_string();
                    let default_512 = "512".to_string();
                    let d_model = layer.parameters.get("d_model").unwrap_or(&default_768);
                    let max_len = layer.parameters.get("max_len").unwrap_or(&default_512);
                    format!(
                        "\n            {}: PositionalEncoding::new({}, {})?,",
                        layer.name, d_model, max_len
                    )
                },
                "transformer_block" => {
                    let default_768 = "768".to_string();
                    let default_12 = "12".to_string();
                    let d_model = layer.parameters.get("d_model").unwrap_or(&default_768);
                    let num_heads = layer.parameters.get("num_heads").unwrap_or(&default_12);
                    format!(
                        "\n            {}: TransformerBlock::new({}, {})?,",
                        layer.name, d_model, num_heads
                    )
                },
                "rnn" => {
                    let default_768 = "768".to_string();
                    let input_size = layer.parameters.get("input_size").unwrap_or(&default_768);
                    let hidden_size = layer.parameters.get("hidden_size").unwrap_or(&default_768);
                    format!(
                        "\n            {}: RNN::new({}, {})?,",
                        layer.name, input_size, hidden_size
                    )
                },
                "lstm" => {
                    let default_768 = "768".to_string();
                    let input_size = layer.parameters.get("input_size").unwrap_or(&default_768);
                    let hidden_size = layer.parameters.get("hidden_size").unwrap_or(&default_768);
                    format!(
                        "\n            {}: LSTM::new({}, {})?,",
                        layer.name, input_size, hidden_size
                    )
                },
                "gru" => {
                    let default_768 = "768".to_string();
                    let input_size = layer.parameters.get("input_size").unwrap_or(&default_768);
                    let hidden_size = layer.parameters.get("hidden_size").unwrap_or(&default_768);
                    format!(
                        "\n            {}: GRU::new({}, {})?,",
                        layer.name, input_size, hidden_size
                    )
                },
                "relu" | "gelu" | "silu" | "tanh" | "sigmoid" | "softmax" => String::new(), // Activation functions don't need initialization
                _ => format!(
                    "\n            // Unsupported layer '{}' - please implement manually",
                    layer.layer_type
                ),
            })
            .collect::<Vec<_>>()
            .join("");

        (layer_fields, layer_init)
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
}
