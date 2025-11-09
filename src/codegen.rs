use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTemplate {
    pub name: String,
    pub architecture: String,
    pub layers: Vec<LayerConfig>,
    pub config: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layer_type: String,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
}

#[derive(Debug, Clone)]
pub struct CodeGenerator {
    templates: HashMap<String, String>,
    output_path: String,
}

impl CodeGenerator {
    pub fn new(output_path: impl Into<String>) -> Self {
        let mut generator = Self {
            templates: HashMap::new(),
            output_path: output_path.into(),
        };

        generator.load_default_templates();
        generator
    }

    fn load_default_templates(&mut self) {
        // Transformer model template
        self.templates.insert("transformer".to_string(), include_str!("../templates/transformer_template.rs").to_string());

        // Extended LSTM (xLSTM) model template - cutting-edge LSTM revival architecture
        self.templates.insert("xlstm".to_string(), include_str!("../templates/xlstm_template.rs").to_string());

        // CNN model template
        self.templates.insert("cnn".to_string(), include_str!("../templates/cnn_template.rs").to_string());

        // Pipeline template
        self.templates.insert("pipeline".to_string(), include_str!("../templates/pipeline_template.rs").to_string());

        // Training loop template
        self.templates.insert("training".to_string(), include_str!("../templates/training_template.rs").to_string());
    }

    pub fn generate_model(&self, template: &ModelTemplate) -> Result<String, Box<dyn std::error::Error>> {
        let template_code = self.templates.get(&template.architecture)
            .ok_or_else(|| format!("Template '{}' not found", template.architecture))?;

        let mut code = template_code.clone();

        // Replace placeholders with actual values
        code = code.replace("{{MODEL_NAME}}", &template.name);
        code = code.replace("{{HIDDEN_SIZE}}", &template.config.hidden_size.to_string());
        code = code.replace("{{NUM_LAYERS}}", &template.config.num_layers.to_string());
        code = code.replace("{{NUM_ATTENTION_HEADS}}", &template.config.num_attention_heads.to_string());
        code = code.replace("{{VOCAB_SIZE}}", &template.config.vocab_size.to_string());
        code = code.replace("{{MAX_POSITION_EMBEDDINGS}}", &template.config.max_position_embeddings.to_string());

        // Generate layer implementations
        let layers_code = self.generate_layers(&template.layers)?;
        code = code.replace("{{LAYERS_IMPL}}", &layers_code);

        Ok(code)
    }

    fn generate_layers(&self, layers: &[LayerConfig]) -> Result<String, Box<dyn std::error::Error>> {
        let mut layers_code = String::new();

        for (idx, layer) in layers.iter().enumerate() {
            match layer.layer_type.as_str() {
                "attention" => {
                    layers_code.push_str(&format!(
                        "        let attention_{idx} = MultiHeadAttention::new(\n            {hidden_size},\n            {num_heads}\n        );\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768))),
                        num_heads = layer.params.get("num_heads").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(12)))
                    ));
                }
                "feedforward" => {
                    layers_code.push_str(&format!(
                        "        let ffn_{idx} = FeedForward::new(\n            {hidden_size},\n            {intermediate_size}\n        );\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768))),
                        intermediate_size = layer.params.get("intermediate_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(3072)))
                    ));
                }
                "layer_norm" => {
                    layers_code.push_str(&format!(
                        "        let layer_norm_{idx} = LayerNorm::new({hidden_size});\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768)))
                    ));
                }
                // Modern layer types for cutting-edge architectures
                "rms_norm" => {
                    layers_code.push_str(&format!(
                        "        let rms_norm_{idx} = RMSNorm::new(\n            {hidden_size},\n            {eps}\n        );\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768))),
                        eps = layer.params.get("eps").unwrap_or(&serde_json::Value::Number(serde_json::Number::from_f64(1e-5).unwrap()))
                    ));
                }
                "swiglu" => {
                    layers_code.push_str(&format!(
                        "        let swiglu_{idx} = SwiGLU::new(\n            {hidden_size},\n            {intermediate_size}\n        );\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768))),
                        intermediate_size = layer.params.get("intermediate_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(3072)))
                    ));
                }
                "rope" => {
                    layers_code.push_str(&format!(
                        "        let rope_{idx} = RotaryPositionalEmbedding::new(\n            {head_dim},\n            {max_position_embeddings},\n            {base}\n        );\n",
                        idx = idx,
                        head_dim = layer.params.get("head_dim").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(64))),
                        max_position_embeddings = layer.params.get("max_position_embeddings").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(2048))),
                        base = layer.params.get("base").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(10000)))
                    ));
                }
                "group_norm" => {
                    layers_code.push_str(&format!(
                        "        let group_norm_{idx} = GroupNorm::new(\n            {num_groups},\n            {num_channels},\n            {eps}\n        );\n",
                        idx = idx,
                        num_groups = layer.params.get("num_groups").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(32))),
                        num_channels = layer.params.get("num_channels").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768))),
                        eps = layer.params.get("eps").unwrap_or(&serde_json::Value::Number(serde_json::Number::from_f64(1e-5).unwrap()))
                    ));
                }
                "slstm" => {
                    layers_code.push_str(&format!(
                        "        let slstm_{idx} = SLSTMBlock::new(\n            {hidden_size},\n            &config\n        );\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768)))
                    ));
                }
                "mlstm" => {
                    layers_code.push_str(&format!(
                        "        let mlstm_{idx} = MLSTMBlock::new(\n            {hidden_size},\n            &config\n        );\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768)))
                    ));
                }
                "ring_attention" => {
                    layers_code.push_str(&format!(
                        "        let ring_attention_{idx} = RingAttention::new(\n            {hidden_size},\n            {num_heads},\n            {chunk_size},\n            {num_devices}\n        );\n",
                        idx = idx,
                        hidden_size = layer.params.get("hidden_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(768))),
                        num_heads = layer.params.get("num_heads").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(12))),
                        chunk_size = layer.params.get("chunk_size").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(4096))),
                        num_devices = layer.params.get("num_devices").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(8)))
                    ));
                }
                _ => {
                    return Err(format!("Unknown layer type: {}. Supported types: attention, feedforward, layer_norm, rms_norm, swiglu, rope, group_norm, slstm, mlstm, ring_attention", layer.layer_type).into());
                }
            }
        }

        Ok(layers_code)
    }

    pub fn generate_pipeline(&self, name: &str, model_type: &str, task: &str) -> Result<String, Box<dyn std::error::Error>> {
        let template_code = self.templates.get("pipeline")
            .ok_or_else(|| "Pipeline template not found")?;

        let mut code = template_code.clone();
        code = code.replace("{{PIPELINE_NAME}}", name);
        code = code.replace("{{MODEL_TYPE}}", model_type);
        code = code.replace("{{TASK_TYPE}}", task);

        Ok(code)
    }

    pub fn generate_training_loop(&self, model_name: &str, optimizer: &str, loss_fn: &str) -> Result<String, Box<dyn std::error::Error>> {
        let template_code = self.templates.get("training")
            .ok_or_else(|| "Training template not found")?;

        let mut code = template_code.clone();
        code = code.replace("{{MODEL_NAME}}", model_name);
        code = code.replace("{{OPTIMIZER}}", optimizer);
        code = code.replace("{{LOSS_FUNCTION}}", loss_fn);

        Ok(code)
    }

    pub fn generate_to_file(&self, content: &str, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new(&self.output_path).join(filename);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&path, content)?;
        println!("Generated code written to: {}", path.display());

        Ok(())
    }

    pub fn list_available_templates(&self) -> Vec<&String> {
        self.templates.keys().collect()
    }

    pub fn add_custom_template(&mut self, name: String, template: String) {
        self.templates.insert(name, template);
    }
}

pub fn create_model_from_config(config_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config_content = fs::read_to_string(config_path)?;
    let template: ModelTemplate = serde_json::from_str(&config_content)?;

    let generator = CodeGenerator::new(output_path);
    let generated_code = generator.generate_model(&template)?;

    let filename = format!("{}_model.rs", template.name.to_lowercase());
    generator.generate_to_file(&generated_code, &filename)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_generator_creation() {
        let generator = CodeGenerator::new("./generated");
        assert_eq!(generator.output_path, "./generated");
        assert!(!generator.templates.is_empty());
    }

    #[test]
    fn test_template_listing() {
        let generator = CodeGenerator::new("./generated");
        let templates = generator.list_available_templates();
        assert!(templates.len() > 0);
    }

    #[test]
    fn test_model_generation() {
        let generator = CodeGenerator::new("./generated");

        let config = ModelConfig {
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            vocab_size: 30000,
            max_position_embeddings: 512,
        };

        let template = ModelTemplate {
            name: "TestModel".to_string(),
            architecture: "transformer".to_string(),
            layers: vec![
                LayerConfig {
                    layer_type: "attention".to_string(),
                    params: {
                        let mut params = HashMap::new();
                        params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("num_heads".to_string(), serde_json::Value::Number(serde_json::Number::from(12)));
                        params
                    },
                },
                LayerConfig {
                    layer_type: "feedforward".to_string(),
                    params: {
                        let mut params = HashMap::new();
                        params.insert("hidden_size".to_string(), serde_json::Value::Number(serde_json::Number::from(768)));
                        params.insert("intermediate_size".to_string(), serde_json::Value::Number(serde_json::Number::from(3072)));
                        params
                    },
                },
            ],
            config,
        };

        let result = generator.generate_model(&template);
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(code.contains("TestModel"));
        assert!(code.contains("768"));
        assert!(code.contains("12"));
    }
}