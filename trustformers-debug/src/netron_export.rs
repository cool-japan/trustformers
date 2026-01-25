//! Netron export functionality for model visualization
//!
//! This module provides tools to export TrustformeRS models to formats compatible with
//! Netron (<https://netron.app/>), a powerful neural network visualizer.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// ONNX-like model representation for Netron visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetronModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Graph definition
    pub graph: ModelGraph,
    /// Model version
    pub version: String,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Model author
    pub author: Option<String>,
    /// Model version
    pub version: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// Model graph containing nodes and edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGraph {
    /// Graph name
    pub name: String,
    /// Input tensors
    pub inputs: Vec<TensorInfo>,
    /// Output tensors
    pub outputs: Vec<TensorInfo>,
    /// Graph nodes (layers/operations)
    pub nodes: Vec<GraphNode>,
    /// Initializers (weights and biases)
    pub initializers: Vec<TensorData>,
}

/// Tensor information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type (e.g., "float32", "int64")
    pub dtype: String,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Optional documentation
    pub doc_string: Option<String>,
}

/// Graph node representing a layer or operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node name
    pub name: String,
    /// Operation type (e.g., "Linear", "Conv2d", "Softmax")
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Node attributes
    pub attributes: HashMap<String, AttributeValue>,
    /// Optional documentation
    pub doc_string: Option<String>,
}

/// Attribute value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// Array of integers
    Ints(Vec<i64>),
    /// Array of floats
    Floats(Vec<f64>),
    /// Array of strings
    Strings(Vec<String>),
}

/// Tensor data for weights and biases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: String,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Raw data (encoded as base64 for binary data)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<f32>>,
    /// Data location (for external data)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_location: Option<String>,
}

/// Netron exporter for model visualization
pub struct NetronExporter {
    model: NetronModel,
    output_format: ExportFormat,
}

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON format (human-readable)
    Json,
    /// ONNX-like binary format
    Onnx,
}

impl NetronExporter {
    /// Create a new Netron exporter
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the model
    /// * `description` - Model description
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::NetronExporter;
    ///
    /// let exporter = NetronExporter::new("bert-base", "BERT base model");
    /// ```
    pub fn new(model_name: &str, description: &str) -> Self {
        let metadata = ModelMetadata {
            name: model_name.to_string(),
            description: description.to_string(),
            author: None,
            version: None,
            license: None,
            properties: HashMap::new(),
        };

        let graph = ModelGraph {
            name: format!("{}_graph", model_name),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            initializers: Vec::new(),
        };

        let model = NetronModel {
            metadata,
            graph,
            version: "1.0".to_string(),
        };

        Self {
            model,
            output_format: ExportFormat::Json,
        }
    }

    /// Set the export format
    pub fn with_format(mut self, format: ExportFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set model metadata
    pub fn set_metadata(&mut self, metadata: ModelMetadata) {
        self.model.metadata = metadata;
    }

    /// Add model author
    pub fn set_author(&mut self, author: &str) {
        self.model.metadata.author = Some(author.to_string());
    }

    /// Add model version
    pub fn set_version(&mut self, version: &str) {
        self.model.metadata.version = Some(version.to_string());
    }

    /// Add a custom property to metadata
    pub fn add_property(&mut self, key: &str, value: &str) {
        self.model.metadata.properties.insert(key.to_string(), value.to_string());
    }

    /// Add an input tensor
    pub fn add_input(&mut self, name: &str, dtype: &str, shape: Vec<i64>) {
        self.model.graph.inputs.push(TensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            doc_string: None,
        });
    }

    /// Add an output tensor
    pub fn add_output(&mut self, name: &str, dtype: &str, shape: Vec<i64>) {
        self.model.graph.outputs.push(TensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            doc_string: None,
        });
    }

    /// Add a graph node (layer/operation)
    ///
    /// # Example
    ///
    /// ```
    /// # use trustformers_debug::NetronExporter;
    /// # use std::collections::HashMap;
    /// let mut exporter = NetronExporter::new("model", "test model");
    ///
    /// let mut attrs = HashMap::new();
    /// attrs.insert("in_features".to_string(),
    ///              trustformers_debug::netron_export::AttributeValue::Int(768));
    /// attrs.insert("out_features".to_string(),
    ///              trustformers_debug::netron_export::AttributeValue::Int(3072));
    ///
    /// exporter.add_node(
    ///     "fc1",
    ///     "Linear",
    ///     vec!["input".to_string()],
    ///     vec!["hidden".to_string()],
    ///     attrs,
    /// );
    /// ```
    pub fn add_node(
        &mut self,
        name: &str,
        op_type: &str,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: HashMap<String, AttributeValue>,
    ) {
        self.model.graph.nodes.push(GraphNode {
            name: name.to_string(),
            op_type: op_type.to_string(),
            inputs,
            outputs,
            attributes,
            doc_string: None,
        });
    }

    /// Add a node with documentation
    pub fn add_node_with_doc(
        &mut self,
        name: &str,
        op_type: &str,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: HashMap<String, AttributeValue>,
        doc_string: &str,
    ) {
        self.model.graph.nodes.push(GraphNode {
            name: name.to_string(),
            op_type: op_type.to_string(),
            inputs,
            outputs,
            attributes,
            doc_string: Some(doc_string.to_string()),
        });
    }

    /// Add tensor data (weights/biases)
    pub fn add_tensor_data(
        &mut self,
        name: &str,
        dtype: &str,
        shape: Vec<i64>,
        data: Option<Vec<f32>>,
    ) {
        self.model.graph.initializers.push(TensorData {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            data,
            data_location: None,
        });
    }

    /// Export the model to a file
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_debug::NetronExporter;
    /// # let exporter = NetronExporter::new("model", "test");
    /// exporter.export("model.json").unwrap();
    /// ```
    pub fn export<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        match self.output_format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&self.model)?;
                fs::write(path, json)?;
            },
            ExportFormat::Onnx => {
                // For now, export as JSON with .onnx extension
                // A full ONNX protobuf implementation would require additional dependencies
                let json = serde_json::to_string_pretty(&self.model)?;
                fs::write(path, json)?;
            },
        }

        Ok(())
    }

    /// Get a reference to the model
    pub fn model(&self) -> &NetronModel {
        &self.model
    }

    /// Get a mutable reference to the model
    pub fn model_mut(&mut self) -> &mut NetronModel {
        &mut self.model
    }

    /// Export model to a string (JSON format)
    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.model)?)
    }

    /// Create a simple linear layer node
    pub fn create_linear_node(
        name: &str,
        input_name: &str,
        output_name: &str,
        in_features: i64,
        out_features: i64,
        has_bias: bool,
    ) -> GraphNode {
        let mut attributes = HashMap::new();
        attributes.insert("in_features".to_string(), AttributeValue::Int(in_features));
        attributes.insert(
            "out_features".to_string(),
            AttributeValue::Int(out_features),
        );
        attributes.insert("bias".to_string(), AttributeValue::Bool(has_bias));

        GraphNode {
            name: name.to_string(),
            op_type: "Linear".to_string(),
            inputs: vec![input_name.to_string()],
            outputs: vec![output_name.to_string()],
            attributes,
            doc_string: None,
        }
    }

    /// Create a transformer attention node
    pub fn create_attention_node(
        name: &str,
        input_name: &str,
        output_name: &str,
        num_heads: i64,
        head_dim: i64,
    ) -> GraphNode {
        let mut attributes = HashMap::new();
        attributes.insert("num_heads".to_string(), AttributeValue::Int(num_heads));
        attributes.insert("head_dim".to_string(), AttributeValue::Int(head_dim));

        GraphNode {
            name: name.to_string(),
            op_type: "MultiHeadAttention".to_string(),
            inputs: vec![input_name.to_string()],
            outputs: vec![output_name.to_string()],
            attributes,
            doc_string: Some("Multi-head self-attention layer".to_string()),
        }
    }

    /// Create a layer normalization node
    pub fn create_layernorm_node(
        name: &str,
        input_name: &str,
        output_name: &str,
        normalized_shape: Vec<i64>,
        eps: f64,
    ) -> GraphNode {
        let mut attributes = HashMap::new();
        attributes.insert(
            "normalized_shape".to_string(),
            AttributeValue::Ints(normalized_shape),
        );
        attributes.insert("eps".to_string(), AttributeValue::Float(eps));

        GraphNode {
            name: name.to_string(),
            op_type: "LayerNorm".to_string(),
            inputs: vec![input_name.to_string()],
            outputs: vec![output_name.to_string()],
            attributes,
            doc_string: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_netron_exporter_creation() {
        let exporter = NetronExporter::new("test_model", "A test model");
        assert_eq!(exporter.model.metadata.name, "test_model");
        assert_eq!(exporter.model.metadata.description, "A test model");
    }

    #[test]
    fn test_add_input_output() {
        let mut exporter = NetronExporter::new("test", "test");

        exporter.add_input("input_ids", "int64", vec![1, 128]);
        exporter.add_output("logits", "float32", vec![1, 128, 30522]);

        assert_eq!(exporter.model.graph.inputs.len(), 1);
        assert_eq!(exporter.model.graph.outputs.len(), 1);
        assert_eq!(exporter.model.graph.inputs[0].name, "input_ids");
    }

    #[test]
    fn test_add_node() {
        let mut exporter = NetronExporter::new("test", "test");

        let mut attrs = HashMap::new();
        attrs.insert("in_features".to_string(), AttributeValue::Int(768));
        attrs.insert("out_features".to_string(), AttributeValue::Int(3072));

        exporter.add_node(
            "fc1",
            "Linear",
            vec!["input".to_string()],
            vec!["output".to_string()],
            attrs,
        );

        assert_eq!(exporter.model.graph.nodes.len(), 1);
        assert_eq!(exporter.model.graph.nodes[0].name, "fc1");
        assert_eq!(exporter.model.graph.nodes[0].op_type, "Linear");
    }

    #[test]
    fn test_export_json() {
        let temp_dir = env::temp_dir();
        let output_path = temp_dir.join("test_model.json");

        let mut exporter = NetronExporter::new("test_model", "Test model");
        exporter.add_input("input", "float32", vec![1, 10]);
        exporter.add_output("output", "float32", vec![1, 5]);

        exporter.export(&output_path).unwrap();
        assert!(output_path.exists());

        // Clean up
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_create_linear_node() {
        let node = NetronExporter::create_linear_node("fc1", "input", "output", 768, 3072, true);

        assert_eq!(node.name, "fc1");
        assert_eq!(node.op_type, "Linear");
        assert!(node.attributes.contains_key("in_features"));
        assert!(node.attributes.contains_key("bias"));
    }

    #[test]
    fn test_create_attention_node() {
        let node = NetronExporter::create_attention_node("attn", "input", "output", 12, 64);

        assert_eq!(node.op_type, "MultiHeadAttention");
        assert!(node.doc_string.is_some());
    }

    #[test]
    fn test_metadata_setters() {
        let mut exporter = NetronExporter::new("test", "test");

        exporter.set_author("Test Author");
        exporter.set_version("1.0.0");
        exporter.add_property("framework", "TrustformeRS");

        assert_eq!(
            exporter.model.metadata.author,
            Some("Test Author".to_string())
        );
        assert_eq!(exporter.model.metadata.version, Some("1.0.0".to_string()));
        assert_eq!(
            exporter.model.metadata.properties.get("framework"),
            Some(&"TrustformeRS".to_string())
        );
    }

    #[test]
    fn test_to_json_string() {
        let mut exporter = NetronExporter::new("test", "test");
        exporter.add_input("input", "float32", vec![1, 10]);

        let json = exporter.to_json_string().unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("input"));
    }

    #[test]
    fn test_add_tensor_data() {
        let mut exporter = NetronExporter::new("test", "test");

        let weights = vec![0.1, 0.2, 0.3, 0.4];
        exporter.add_tensor_data("layer.weight", "float32", vec![2, 2], Some(weights));

        assert_eq!(exporter.model.graph.initializers.len(), 1);
        assert_eq!(exporter.model.graph.initializers[0].name, "layer.weight");
    }
}
