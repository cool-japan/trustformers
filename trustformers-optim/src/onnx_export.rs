//! ONNX Optimizer Export
//!
//! This module provides functionality to export optimizer configurations and states
//! to ONNX format, enabling deployment and optimization in ONNX Runtime environments.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// ONNX Graph Node representation for optimizer operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, Value>,
}

/// ONNX Graph representation for optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXGraph {
    pub name: String,
    pub nodes: Vec<ONNXNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub initializers: HashMap<String, Vec<f32>>,
}

/// ONNX Model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXModel {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub graph: ONNXGraph,
}

/// Optimizer configuration for ONNX export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f32,
    pub parameters: HashMap<String, Value>,
}

/// ONNX Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXExportConfig {
    pub model_name: String,
    pub opset_version: i64,
    pub export_params: bool,
    pub export_raw_ir: bool,
    pub keep_initializers_as_inputs: bool,
    pub custom_opsets: HashMap<String, i64>,
    pub verbose: bool,
}

impl Default for ONNXExportConfig {
    fn default() -> Self {
        Self {
            model_name: "TrustformeRS_Optimizer".to_string(),
            opset_version: 17,
            export_params: true,
            export_raw_ir: false,
            keep_initializers_as_inputs: false,
            custom_opsets: HashMap::new(),
            verbose: false,
        }
    }
}

/// ONNX Optimizer metadata for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXOptimizerMetadata {
    pub optimizer_type: String,
    pub version: String,
    pub hyperparameters: HashMap<String, Value>,
    pub state_variables: Vec<String>,
    pub export_timestamp: String,
    pub framework_version: String,
}

impl Default for ONNXOptimizerMetadata {
    fn default() -> Self {
        Self {
            optimizer_type: "Adam".to_string(),
            version: "1.0".to_string(),
            hyperparameters: HashMap::new(),
            state_variables: Vec::new(),
            export_timestamp: "2025-07-22T00:00:00Z".to_string(),
            framework_version: "0.1.0".to_string(),
        }
    }
}

/// ONNX Optimizer Exporter
pub struct ONNXOptimizerExporter {
    producer_name: String,
    producer_version: String,
}

impl ONNXOptimizerExporter {
    /// Create a new ONNX optimizer exporter
    pub fn new() -> Self {
        Self {
            producer_name: "TrustformeRS".to_string(),
            producer_version: "1.0.0".to_string(),
        }
    }

    /// Export Adam optimizer to ONNX format
    pub fn export_adam(
        &self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<ONNXModel> {
        let mut nodes = Vec::new();
        let mut initializers = HashMap::new();

        // Add learning rate as initializer
        initializers.insert("learning_rate".to_string(), vec![learning_rate]);
        initializers.insert("beta1".to_string(), vec![beta1]);
        initializers.insert("beta2".to_string(), vec![beta2]);
        initializers.insert("epsilon".to_string(), vec![epsilon]);
        initializers.insert("weight_decay".to_string(), vec![weight_decay]);

        // Create Adam optimizer node
        let mut adam_attrs = HashMap::new();
        adam_attrs.insert(
            "alpha".to_string(),
            Value::Number(
                serde_json::Number::from_f64(learning_rate as f64)
                    .expect("Invalid learning_rate: not a finite number"),
            ),
        );
        adam_attrs.insert(
            "beta".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta1 as f64)
                    .expect("Invalid beta1: not a finite number"),
            ),
        );
        adam_attrs.insert(
            "beta2".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta2 as f64)
                    .expect("Invalid beta2: not a finite number"),
            ),
        );
        adam_attrs.insert(
            "epsilon".to_string(),
            Value::Number(
                serde_json::Number::from_f64(epsilon as f64)
                    .expect("Invalid epsilon: not a finite number"),
            ),
        );
        adam_attrs.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .expect("Invalid weight_decay: not a finite number"),
            ),
        );

        let adam_node = ONNXNode {
            name: "adam_optimizer".to_string(),
            op_type: "Adam".to_string(),
            inputs: vec![
                "gradients".to_string(),
                "learning_rate".to_string(),
                "beta1".to_string(),
                "beta2".to_string(),
                "epsilon".to_string(),
                "weight_decay".to_string(),
            ],
            outputs: vec!["updated_parameters".to_string()],
            attributes: adam_attrs,
        };

        nodes.push(adam_node);

        let graph = ONNXGraph {
            name: "adam_optimizer_graph".to_string(),
            nodes,
            inputs: vec!["gradients".to_string()],
            outputs: vec!["updated_parameters".to_string()],
            initializers,
        };

        Ok(ONNXModel {
            ir_version: 7,
            producer_name: self.producer_name.clone(),
            producer_version: self.producer_version.clone(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            graph,
        })
    }

    /// Export SGD optimizer to ONNX format
    pub fn export_sgd(
        &self,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Result<ONNXModel> {
        let mut nodes = Vec::new();
        let mut initializers = HashMap::new();

        // Add hyperparameters as initializers
        initializers.insert("learning_rate".to_string(), vec![learning_rate]);
        initializers.insert("momentum".to_string(), vec![momentum]);
        initializers.insert("weight_decay".to_string(), vec![weight_decay]);

        // Create SGD optimizer node
        let mut sgd_attrs = HashMap::new();
        sgd_attrs.insert(
            "learning_rate".to_string(),
            Value::Number(
                serde_json::Number::from_f64(learning_rate as f64)
                    .expect("Invalid learning_rate: not a finite number"),
            ),
        );
        sgd_attrs.insert(
            "momentum".to_string(),
            Value::Number(
                serde_json::Number::from_f64(momentum as f64)
                    .expect("Invalid momentum: not a finite number"),
            ),
        );
        sgd_attrs.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .expect("Invalid weight_decay: not a finite number"),
            ),
        );
        sgd_attrs.insert("nesterov".to_string(), Value::Bool(nesterov));

        let sgd_node = ONNXNode {
            name: "sgd_optimizer".to_string(),
            op_type: "SGD".to_string(),
            inputs: vec![
                "gradients".to_string(),
                "learning_rate".to_string(),
                "momentum".to_string(),
                "weight_decay".to_string(),
            ],
            outputs: vec!["updated_parameters".to_string()],
            attributes: sgd_attrs,
        };

        nodes.push(sgd_node);

        let graph = ONNXGraph {
            name: "sgd_optimizer_graph".to_string(),
            nodes,
            inputs: vec!["gradients".to_string()],
            outputs: vec!["updated_parameters".to_string()],
            initializers,
        };

        Ok(ONNXModel {
            ir_version: 7,
            producer_name: self.producer_name.clone(),
            producer_version: self.producer_version.clone(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            graph,
        })
    }

    /// Export AdamW optimizer to ONNX format
    pub fn export_adamw(
        &self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<ONNXModel> {
        let mut nodes = Vec::new();
        let mut initializers = HashMap::new();

        // Add hyperparameters as initializers
        initializers.insert("learning_rate".to_string(), vec![learning_rate]);
        initializers.insert("beta1".to_string(), vec![beta1]);
        initializers.insert("beta2".to_string(), vec![beta2]);
        initializers.insert("epsilon".to_string(), vec![epsilon]);
        initializers.insert("weight_decay".to_string(), vec![weight_decay]);

        // Create AdamW optimizer node
        let mut adamw_attrs = HashMap::new();
        adamw_attrs.insert(
            "alpha".to_string(),
            Value::Number(
                serde_json::Number::from_f64(learning_rate as f64)
                    .expect("Invalid learning_rate: not a finite number"),
            ),
        );
        adamw_attrs.insert(
            "beta".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta1 as f64)
                    .expect("Invalid beta1: not a finite number"),
            ),
        );
        adamw_attrs.insert(
            "beta2".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta2 as f64)
                    .expect("Invalid beta2: not a finite number"),
            ),
        );
        adamw_attrs.insert(
            "epsilon".to_string(),
            Value::Number(
                serde_json::Number::from_f64(epsilon as f64)
                    .expect("Invalid epsilon: not a finite number"),
            ),
        );
        adamw_attrs.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .expect("Invalid weight_decay: not a finite number"),
            ),
        );

        let adamw_node = ONNXNode {
            name: "adamw_optimizer".to_string(),
            op_type: "AdamW".to_string(),
            inputs: vec![
                "gradients".to_string(),
                "learning_rate".to_string(),
                "beta1".to_string(),
                "beta2".to_string(),
                "epsilon".to_string(),
                "weight_decay".to_string(),
            ],
            outputs: vec!["updated_parameters".to_string()],
            attributes: adamw_attrs,
        };

        nodes.push(adamw_node);

        let graph = ONNXGraph {
            name: "adamw_optimizer_graph".to_string(),
            nodes,
            inputs: vec!["gradients".to_string()],
            outputs: vec!["updated_parameters".to_string()],
            initializers,
        };

        Ok(ONNXModel {
            ir_version: 7,
            producer_name: self.producer_name.clone(),
            producer_version: self.producer_version.clone(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            graph,
        })
    }

    /// Export optimizer configuration to JSON format for ONNX metadata
    pub fn export_config(&self, config: &OptimizerConfig) -> Result<String> {
        serde_json::to_string_pretty(config)
            .map_err(|e| anyhow!("Failed to serialize optimizer config: {}", e))
    }

    /// Save ONNX model to file
    pub fn save_model(&self, model: &ONNXModel, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(model)
            .map_err(|e| anyhow!("Failed to serialize ONNX model: {}", e))?;

        std::fs::write(path, json)
            .map_err(|e| anyhow!("Failed to write ONNX model to file: {}", e))?;

        Ok(())
    }

    /// Create optimizer config from common optimizers
    pub fn create_adam_config(
        &self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> OptimizerConfig {
        let mut parameters = HashMap::new();
        parameters.insert(
            "beta1".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta1 as f64)
                    .expect("Invalid beta1: not a finite number"),
            ),
        );
        parameters.insert(
            "beta2".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta2 as f64)
                    .expect("Invalid beta2: not a finite number"),
            ),
        );
        parameters.insert(
            "epsilon".to_string(),
            Value::Number(
                serde_json::Number::from_f64(epsilon as f64)
                    .expect("Invalid epsilon: not a finite number"),
            ),
        );
        parameters.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .expect("Invalid weight_decay: not a finite number"),
            ),
        );

        OptimizerConfig {
            optimizer_type: "Adam".to_string(),
            learning_rate,
            parameters,
        }
    }

    pub fn create_sgd_config(
        &self,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> OptimizerConfig {
        let mut parameters = HashMap::new();
        parameters.insert(
            "momentum".to_string(),
            Value::Number(
                serde_json::Number::from_f64(momentum as f64)
                    .expect("Invalid momentum: not a finite number"),
            ),
        );
        parameters.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .expect("Invalid weight_decay: not a finite number"),
            ),
        );
        parameters.insert("nesterov".to_string(), Value::Bool(nesterov));

        OptimizerConfig {
            optimizer_type: "SGD".to_string(),
            learning_rate,
            parameters,
        }
    }

    pub fn create_adamw_config(
        &self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> OptimizerConfig {
        let mut parameters = HashMap::new();
        parameters.insert(
            "beta1".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta1 as f64)
                    .expect("Invalid beta1: not a finite number"),
            ),
        );
        parameters.insert(
            "beta2".to_string(),
            Value::Number(
                serde_json::Number::from_f64(beta2 as f64)
                    .expect("Invalid beta2: not a finite number"),
            ),
        );
        parameters.insert(
            "epsilon".to_string(),
            Value::Number(
                serde_json::Number::from_f64(epsilon as f64)
                    .expect("Invalid epsilon: not a finite number"),
            ),
        );
        parameters.insert(
            "weight_decay".to_string(),
            Value::Number(
                serde_json::Number::from_f64(weight_decay as f64)
                    .expect("Invalid weight_decay: not a finite number"),
            ),
        );

        OptimizerConfig {
            optimizer_type: "AdamW".to_string(),
            learning_rate,
            parameters,
        }
    }
}

impl Default for ONNXOptimizerExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for ONNX export
pub mod utils {
    use super::*;

    /// Validate ONNX model structure
    pub fn validate_model(model: &ONNXModel) -> Result<()> {
        if model.graph.nodes.is_empty() {
            return Err(anyhow!("ONNX model must have at least one node"));
        }

        if model.graph.inputs.is_empty() {
            return Err(anyhow!("ONNX model must have at least one input"));
        }

        if model.graph.outputs.is_empty() {
            return Err(anyhow!("ONNX model must have at least one output"));
        }

        // Validate node connections
        for node in &model.graph.nodes {
            for input in &node.inputs {
                if !model.graph.inputs.contains(input)
                    && !model.graph.initializers.contains_key(input)
                {
                    // Check if input is output of another node
                    let is_node_output =
                        model.graph.nodes.iter().any(|n| n.outputs.contains(input));

                    if !is_node_output {
                        return Err(anyhow!("Node input '{}' is not connected", input));
                    }
                }
            }
        }

        Ok(())
    }

    /// Create ONNX model with learning rate scheduler
    pub fn create_with_scheduler(
        optimizer_model: ONNXModel,
        schedule_type: &str,
        schedule_params: HashMap<String, f32>,
    ) -> Result<ONNXModel> {
        let mut model = optimizer_model;

        // Add scheduler node
        let mut scheduler_attrs = HashMap::new();
        for (key, value) in schedule_params {
            scheduler_attrs.insert(
                key,
                Value::Number(
                    serde_json::Number::from_f64(value as f64)
                        .expect("Invalid value: not a finite number"),
                ),
            );
        }

        let scheduler_node = ONNXNode {
            name: "lr_scheduler".to_string(),
            op_type: schedule_type.to_string(),
            inputs: vec!["step".to_string()],
            outputs: vec!["scheduled_learning_rate".to_string()],
            attributes: scheduler_attrs,
        };

        model.graph.nodes.insert(0, scheduler_node);
        model.graph.inputs.push("step".to_string());

        // Update optimizer node to use scheduled learning rate
        if let Some(optimizer_node) = model
            .graph
            .nodes
            .iter_mut()
            .find(|n| n.op_type == "Adam" || n.op_type == "SGD" || n.op_type == "AdamW")
        {
            if let Some(lr_input_idx) =
                optimizer_node.inputs.iter().position(|i| i == "learning_rate")
            {
                optimizer_node.inputs[lr_input_idx] = "scheduled_learning_rate".to_string();
            }
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_adam_export() {
        let exporter = ONNXOptimizerExporter::new();
        let model = exporter.export_adam(0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();

        assert_eq!(model.graph.name, "adam_optimizer_graph");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.nodes[0].op_type, "Adam");

        utils::validate_model(&model).unwrap();
    }

    #[test]
    fn test_onnx_sgd_export() {
        let exporter = ONNXOptimizerExporter::new();
        let model = exporter.export_sgd(0.01, 0.9, 1e-4, true).unwrap();

        assert_eq!(model.graph.name, "sgd_optimizer_graph");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.nodes[0].op_type, "SGD");

        utils::validate_model(&model).unwrap();
    }

    #[test]
    fn test_onnx_adamw_export() {
        let exporter = ONNXOptimizerExporter::new();
        let model = exporter.export_adamw(0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();

        assert_eq!(model.graph.name, "adamw_optimizer_graph");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.nodes[0].op_type, "AdamW");

        utils::validate_model(&model).unwrap();
    }

    #[test]
    fn test_config_creation() {
        let exporter = ONNXOptimizerExporter::new();

        let adam_config = exporter.create_adam_config(0.001, 0.9, 0.999, 1e-8, 0.01);
        assert_eq!(adam_config.optimizer_type, "Adam");
        assert_eq!(adam_config.learning_rate, 0.001);

        let sgd_config = exporter.create_sgd_config(0.01, 0.9, 1e-4, true);
        assert_eq!(sgd_config.optimizer_type, "SGD");
        assert_eq!(sgd_config.learning_rate, 0.01);
    }

    #[test]
    fn test_config_serialization() {
        let exporter = ONNXOptimizerExporter::new();
        let config = exporter.create_adam_config(0.001, 0.9, 0.999, 1e-8, 0.01);

        let json = exporter.export_config(&config).unwrap();
        assert!(json.contains("Adam"));
        assert!(json.contains("0.001"));
    }

    #[test]
    fn test_model_validation() {
        let exporter = ONNXOptimizerExporter::new();
        let model = exporter.export_adam(0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();

        // Should pass validation
        utils::validate_model(&model).unwrap();

        // Test invalid model
        let mut invalid_model = model.clone();
        invalid_model.graph.nodes.clear();
        assert!(utils::validate_model(&invalid_model).is_err());
    }

    #[test]
    fn test_scheduler_integration() {
        let exporter = ONNXOptimizerExporter::new();
        let base_model = exporter.export_adam(0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();

        let mut schedule_params = HashMap::new();
        schedule_params.insert("decay_rate".to_string(), 0.95);

        let model_with_scheduler =
            utils::create_with_scheduler(base_model, "ExponentialDecay", schedule_params).unwrap();

        assert_eq!(model_with_scheduler.graph.nodes.len(), 2);
        assert_eq!(
            model_with_scheduler.graph.nodes[0].op_type,
            "ExponentialDecay"
        );

        utils::validate_model(&model_with_scheduler).unwrap();
    }
}
