//! Core ML Export for Apple Devices
//!
//! This module provides export capabilities to Apple's Core ML format,
//! enabling optimized inference on iOS, iPadOS, macOS devices with
//! Neural Engine acceleration.
//!
//! Key features:
//! - Transformer model conversion to Core ML format
//! - Neural Engine optimization
//! - Multi-head attention mapping
//! - Quantization support (FP16, INT8)
//! - Metadata and model packaging

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Custom error type for Core ML export operations
#[derive(Debug, Clone)]
pub struct CoreMLError {
    message: String,
}

impl CoreMLError {
    fn new(msg: &str) -> Self {
        Self {
            message: msg.to_string(),
        }
    }
}

impl From<CoreMLError> for JsValue {
    fn from(err: CoreMLError) -> Self {
        JsValue::from_str(&err.message)
    }
}

/// Core ML model format version
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMLVersion {
    /// Core ML 1 (iOS 11+)
    V1,
    /// Core ML 2 (iOS 12+)
    V2,
    /// Core ML 3 (iOS 13+)
    V3,
    /// Core ML 4 (iOS 14+)
    V4,
    /// Core ML 5 (iOS 15+)
    V5,
    /// Core ML 6 (iOS 16+)
    V6,
    /// Core ML 7 (iOS 17+)
    V7,
}

/// Compute unit for Core ML inference
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMLComputeUnit {
    /// CPU only
    CPUOnly,
    /// CPU and GPU
    CPUAndGPU,
    /// CPU and Neural Engine
    CPUAndNeuralEngine,
    /// All available units
    All,
}

/// Precision for Core ML models
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMLPrecision {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point (recommended for Neural Engine)
    Float16,
    /// 8-bit integer
    Int8,
    /// Mixed precision (FP16 for compute, FP32 for storage)
    Mixed,
}

/// Core ML export configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CoreMLExportConfig {
    version: CoreMLVersion,
    compute_unit: CoreMLComputeUnit,
    precision: CoreMLPrecision,
    optimize_for_neural_engine: bool,
    enable_flexible_shapes: bool,
    batch_size: usize,
    sequence_length: usize,
}

#[wasm_bindgen]
impl CoreMLExportConfig {
    /// Create a new Core ML export configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for iPhone
    pub fn iphone() -> Self {
        Self {
            version: CoreMLVersion::V6,
            compute_unit: CoreMLComputeUnit::CPUAndNeuralEngine,
            precision: CoreMLPrecision::Float16,
            optimize_for_neural_engine: true,
            enable_flexible_shapes: false,
            batch_size: 1,
            sequence_length: 512,
        }
    }

    /// Create configuration optimized for iPad
    pub fn ipad() -> Self {
        Self {
            version: CoreMLVersion::V6,
            compute_unit: CoreMLComputeUnit::All,
            precision: CoreMLPrecision::Float16,
            optimize_for_neural_engine: true,
            enable_flexible_shapes: true,
            batch_size: 1,
            sequence_length: 1024,
        }
    }

    /// Create configuration optimized for Mac (Apple Silicon)
    pub fn mac() -> Self {
        Self {
            version: CoreMLVersion::V7,
            compute_unit: CoreMLComputeUnit::All,
            precision: CoreMLPrecision::Mixed,
            optimize_for_neural_engine: true,
            enable_flexible_shapes: true,
            batch_size: 4,
            sequence_length: 2048,
        }
    }

    /// Set Core ML version
    pub fn set_version(&mut self, version: CoreMLVersion) {
        self.version = version;
    }

    /// Set compute unit
    pub fn set_compute_unit(&mut self, unit: CoreMLComputeUnit) {
        self.compute_unit = unit;
    }

    /// Set precision
    pub fn set_precision(&mut self, precision: CoreMLPrecision) {
        self.precision = precision;
    }

    /// Enable Neural Engine optimization
    pub fn enable_neural_engine(&mut self) {
        self.optimize_for_neural_engine = true;
    }

    /// Get current version
    pub fn version(&self) -> CoreMLVersion {
        self.version
    }

    /// Get compute unit
    pub fn compute_unit(&self) -> CoreMLComputeUnit {
        self.compute_unit
    }

    /// Get precision
    pub fn precision(&self) -> CoreMLPrecision {
        self.precision
    }
}

impl Default for CoreMLExportConfig {
    fn default() -> Self {
        Self {
            version: CoreMLVersion::V6,
            compute_unit: CoreMLComputeUnit::CPUAndNeuralEngine,
            precision: CoreMLPrecision::Float16,
            optimize_for_neural_engine: true,
            enable_flexible_shapes: false,
            batch_size: 1,
            sequence_length: 512,
        }
    }
}

/// Core ML model exporter
pub struct CoreMLExporter {
    config: CoreMLExportConfig,
    model_metadata: CoreMLModelMetadata,
    layers: Vec<CoreMLLayer>,
}

impl CoreMLExporter {
    /// Create a new Core ML exporter
    pub fn new(config: CoreMLExportConfig) -> Self {
        Self {
            config,
            model_metadata: CoreMLModelMetadata::default(),
            layers: Vec::new(),
        }
    }

    /// Set model metadata
    pub fn set_metadata(&mut self, metadata: CoreMLModelMetadata) {
        self.model_metadata = metadata;
    }

    /// Add a layer to the model
    pub fn add_layer(&mut self, layer: CoreMLLayer) {
        self.layers.push(layer);
    }

    /// Export transformer model to Core ML format
    pub fn export_model(&self) -> Result<CoreMLModel, JsValue> {
        self.export_model_internal().map_err(|e| e.into())
    }

    /// Test-only version that doesn't require wasm environment
    #[cfg(test)]
    pub fn export_model_test(&self) -> Result<CoreMLModel, CoreMLError> {
        self.export_model_internal()
    }

    fn export_model_internal(&self) -> Result<CoreMLModel, CoreMLError> {
        // Validate model structure
        self.validate_model()?;

        // Convert layers to Core ML operations
        let operations = self.convert_layers_to_operations()?;

        // Apply Neural Engine optimizations
        let optimized_ops = if self.config.optimize_for_neural_engine {
            self.optimize_for_neural_engine(&operations)?
        } else {
            operations
        };

        Ok(CoreMLModel {
            version: self.config.version,
            metadata: self.model_metadata.clone(),
            operations: optimized_ops,
            inputs: self.generate_input_spec(),
            outputs: self.generate_output_spec(),
            compute_unit: self.config.compute_unit,
        })
    }

    fn validate_model(&self) -> Result<(), CoreMLError> {
        if self.layers.is_empty() {
            return Err(CoreMLError::new("Model has no layers"));
        }

        Ok(())
    }

    fn convert_layers_to_operations(&self) -> Result<Vec<CoreMLOperation>, CoreMLError> {
        let mut operations = Vec::new();

        for layer in &self.layers {
            match layer.layer_type {
                CoreMLLayerType::Embedding => {
                    operations.push(CoreMLOperation {
                        op_type: "embedding".to_string(),
                        name: layer.name.clone(),
                        inputs: layer.inputs.clone(),
                        outputs: layer.outputs.clone(),
                        parameters: layer.parameters.clone(),
                    });
                },
                CoreMLLayerType::MultiHeadAttention => {
                    // Multi-head attention requires multiple operations
                    operations.extend(self.convert_mha_to_operations(layer)?);
                },
                CoreMLLayerType::FeedForward => {
                    operations.extend(self.convert_ffn_to_operations(layer)?);
                },
                CoreMLLayerType::LayerNorm => {
                    operations.push(CoreMLOperation {
                        op_type: "layer_norm".to_string(),
                        name: layer.name.clone(),
                        inputs: layer.inputs.clone(),
                        outputs: layer.outputs.clone(),
                        parameters: layer.parameters.clone(),
                    });
                },
                CoreMLLayerType::Linear => {
                    operations.push(CoreMLOperation {
                        op_type: "inner_product".to_string(),
                        name: layer.name.clone(),
                        inputs: layer.inputs.clone(),
                        outputs: layer.outputs.clone(),
                        parameters: layer.parameters.clone(),
                    });
                },
            }
        }

        Ok(operations)
    }

    fn convert_mha_to_operations(
        &self,
        layer: &CoreMLLayer,
    ) -> Result<Vec<CoreMLOperation>, CoreMLError> {
        let mut ops = Vec::new();

        // Q, K, V projections
        ops.push(CoreMLOperation {
            op_type: "inner_product".to_string(),
            name: format!("{}_q_proj", layer.name),
            inputs: layer.inputs.clone(),
            outputs: vec![format!("{}_q", layer.name)],
            parameters: HashMap::new(),
        });

        ops.push(CoreMLOperation {
            op_type: "inner_product".to_string(),
            name: format!("{}_k_proj", layer.name),
            inputs: layer.inputs.clone(),
            outputs: vec![format!("{}_k", layer.name)],
            parameters: HashMap::new(),
        });

        ops.push(CoreMLOperation {
            op_type: "inner_product".to_string(),
            name: format!("{}_v_proj", layer.name),
            inputs: layer.inputs.clone(),
            outputs: vec![format!("{}_v", layer.name)],
            parameters: HashMap::new(),
        });

        // Attention computation
        ops.push(CoreMLOperation {
            op_type: "scaled_dot_product_attention".to_string(),
            name: format!("{}_attention", layer.name),
            inputs: vec![
                format!("{}_q", layer.name),
                format!("{}_k", layer.name),
                format!("{}_v", layer.name),
            ],
            outputs: vec![format!("{}_attn_out", layer.name)],
            parameters: HashMap::new(),
        });

        // Output projection
        ops.push(CoreMLOperation {
            op_type: "inner_product".to_string(),
            name: format!("{}_out_proj", layer.name),
            inputs: vec![format!("{}_attn_out", layer.name)],
            outputs: layer.outputs.clone(),
            parameters: HashMap::new(),
        });

        Ok(ops)
    }

    fn convert_ffn_to_operations(
        &self,
        layer: &CoreMLLayer,
    ) -> Result<Vec<CoreMLOperation>, CoreMLError> {
        let mut ops = Vec::new();

        // First linear layer
        ops.push(CoreMLOperation {
            op_type: "inner_product".to_string(),
            name: format!("{}_fc1", layer.name),
            inputs: layer.inputs.clone(),
            outputs: vec![format!("{}_fc1_out", layer.name)],
            parameters: HashMap::new(),
        });

        // Activation (GELU)
        ops.push(CoreMLOperation {
            op_type: "gelu".to_string(),
            name: format!("{}_gelu", layer.name),
            inputs: vec![format!("{}_fc1_out", layer.name)],
            outputs: vec![format!("{}_gelu_out", layer.name)],
            parameters: HashMap::new(),
        });

        // Second linear layer
        ops.push(CoreMLOperation {
            op_type: "inner_product".to_string(),
            name: format!("{}_fc2", layer.name),
            inputs: vec![format!("{}_gelu_out", layer.name)],
            outputs: layer.outputs.clone(),
            parameters: HashMap::new(),
        });

        Ok(ops)
    }

    fn optimize_for_neural_engine(
        &self,
        operations: &[CoreMLOperation],
    ) -> Result<Vec<CoreMLOperation>, CoreMLError> {
        let mut optimized = Vec::new();

        for op in operations {
            let mut optimized_op = op.clone();

            // Neural Engine optimizations
            match op.op_type.as_str() {
                "inner_product" => {
                    // Ensure weights are in Neural Engine-friendly format
                    optimized_op
                        .parameters
                        .insert("use_neural_engine".to_string(), "true".to_string());
                },
                "layer_norm" => {
                    // Use fused layer norm when possible
                    optimized_op.parameters.insert("fused".to_string(), "true".to_string());
                },
                "gelu" => {
                    // Use approximated GELU for Neural Engine
                    if self.config.precision == CoreMLPrecision::Float16 {
                        optimized_op
                            .parameters
                            .insert("mode".to_string(), "APPROXIMATE".to_string());
                    }
                },
                _ => {},
            }

            optimized.push(optimized_op);
        }

        Ok(optimized)
    }

    fn generate_input_spec(&self) -> Vec<CoreMLTensorSpec> {
        vec![CoreMLTensorSpec {
            name: "input_ids".to_string(),
            shape: vec![
                self.config.batch_size as i64,
                self.config.sequence_length as i64,
            ],
            dtype: "Int32".to_string(),
        }]
    }

    fn generate_output_spec(&self) -> Vec<CoreMLTensorSpec> {
        vec![CoreMLTensorSpec {
            name: "logits".to_string(),
            shape: vec![
                self.config.batch_size as i64,
                self.config.sequence_length as i64,
                -1,
            ],
            dtype: match self.config.precision {
                CoreMLPrecision::Float32 | CoreMLPrecision::Mixed => "Float32".to_string(),
                CoreMLPrecision::Float16 => "Float16".to_string(),
                CoreMLPrecision::Int8 => "Int8".to_string(),
            },
        }]
    }
}

/// Core ML model metadata
#[derive(Debug, Clone, Default)]
pub struct CoreMLModelMetadata {
    pub author: String,
    pub license: String,
    pub description: String,
    pub version: String,
}

/// Core ML layer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLLayerType {
    Embedding,
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    Linear,
}

/// Core ML layer representation
#[derive(Debug, Clone)]
pub struct CoreMLLayer {
    pub name: String,
    pub layer_type: CoreMLLayerType,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub parameters: HashMap<String, String>,
}

/// Core ML operation
#[derive(Debug, Clone)]
pub struct CoreMLOperation {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub parameters: HashMap<String, String>,
}

/// Core ML tensor specification
#[derive(Debug, Clone)]
pub struct CoreMLTensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

/// Complete Core ML model
#[derive(Debug, Clone)]
pub struct CoreMLModel {
    pub version: CoreMLVersion,
    pub metadata: CoreMLModelMetadata,
    pub operations: Vec<CoreMLOperation>,
    pub inputs: Vec<CoreMLTensorSpec>,
    pub outputs: Vec<CoreMLTensorSpec>,
    pub compute_unit: CoreMLComputeUnit,
}

impl CoreMLModel {
    /// Serialize model to ML Model format (simplified)
    pub fn to_mlmodel_json(&self) -> String {
        // In a real implementation, this would generate the actual .mlmodel protobuf format
        // For now, we generate a JSON representation
        format!(
            r#"{{
    "version": "{:?}",
    "compute_unit": "{:?}",
    "inputs": {:?},
    "outputs": {:?},
    "operations_count": {}
}}"#,
            self.version,
            self.compute_unit,
            self.inputs.len(),
            self.outputs.len(),
            self.operations.len()
        )
    }

    /// Estimate model size in bytes
    pub fn estimate_size(&self) -> usize {
        // Simple estimation based on operations count
        self.operations.len() * 1024 * 1024 // ~1MB per operation as rough estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_export_config_default() {
        let config = CoreMLExportConfig::default();
        assert_eq!(config.version(), CoreMLVersion::V6);
        assert_eq!(config.compute_unit(), CoreMLComputeUnit::CPUAndNeuralEngine);
        assert_eq!(config.precision(), CoreMLPrecision::Float16);
    }

    #[test]
    fn test_iphone_config() {
        let config = CoreMLExportConfig::iphone();
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.sequence_length, 512);
        assert!(config.optimize_for_neural_engine);
    }

    #[test]
    fn test_ipad_config() {
        let config = CoreMLExportConfig::ipad();
        assert_eq!(config.sequence_length, 1024);
        assert!(config.enable_flexible_shapes);
    }

    #[test]
    fn test_mac_config() {
        let config = CoreMLExportConfig::mac();
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.sequence_length, 2048);
        assert_eq!(config.precision(), CoreMLPrecision::Mixed);
    }

    #[test]
    fn test_coreml_exporter_creation() {
        let config = CoreMLExportConfig::default();
        let exporter = CoreMLExporter::new(config);
        assert_eq!(exporter.layers.len(), 0);
    }

    #[test]
    fn test_add_layer() {
        let config = CoreMLExportConfig::default();
        let mut exporter = CoreMLExporter::new(config);

        let layer = CoreMLLayer {
            name: "embedding".to_string(),
            layer_type: CoreMLLayerType::Embedding,
            inputs: vec!["input_ids".to_string()],
            outputs: vec!["embeddings".to_string()],
            parameters: HashMap::new(),
        };

        exporter.add_layer(layer);
        assert_eq!(exporter.layers.len(), 1);
    }

    #[test]
    fn test_export_empty_model_fails() {
        let config = CoreMLExportConfig::default();
        let exporter = CoreMLExporter::new(config);

        let result = exporter.export_model_test();
        // Should return an error for empty model
        match result {
            Err(_) => {}, // Expected
            Ok(_) => panic!("Expected error for empty model"),
        }
    }

    #[test]
    fn test_export_with_layers() {
        let config = CoreMLExportConfig::default();
        let mut exporter = CoreMLExporter::new(config);

        exporter.add_layer(CoreMLLayer {
            name: "embedding".to_string(),
            layer_type: CoreMLLayerType::Embedding,
            inputs: vec!["input_ids".to_string()],
            outputs: vec!["embeddings".to_string()],
            parameters: HashMap::new(),
        });

        let result = exporter.export_model_test();
        assert!(result.is_ok());
    }

    #[test]
    fn test_mlmodel_json_generation() {
        let model = CoreMLModel {
            version: CoreMLVersion::V6,
            metadata: CoreMLModelMetadata::default(),
            operations: vec![],
            inputs: vec![],
            outputs: vec![],
            compute_unit: CoreMLComputeUnit::CPUAndNeuralEngine,
        };

        let json = model.to_mlmodel_json();
        assert!(json.contains("V6"));
        assert!(json.contains("CPUAndNeuralEngine"));
    }

    #[test]
    fn test_model_size_estimation() {
        let model = CoreMLModel {
            version: CoreMLVersion::V6,
            metadata: CoreMLModelMetadata::default(),
            operations: vec![CoreMLOperation {
                op_type: "embedding".to_string(),
                name: "emb".to_string(),
                inputs: vec![],
                outputs: vec![],
                parameters: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            compute_unit: CoreMLComputeUnit::All,
        };

        let size = model.estimate_size();
        assert!(size > 0);
    }

    #[test]
    fn test_coreml_version_comparison() {
        assert_ne!(CoreMLVersion::V6, CoreMLVersion::V7);
        assert_eq!(CoreMLVersion::V6, CoreMLVersion::V6);
    }

    #[test]
    fn test_compute_unit_types() {
        let units = [
            CoreMLComputeUnit::CPUOnly,
            CoreMLComputeUnit::CPUAndGPU,
            CoreMLComputeUnit::CPUAndNeuralEngine,
            CoreMLComputeUnit::All,
        ];

        assert_eq!(units.len(), 4);
    }

    #[test]
    fn test_precision_types() {
        let precisions = [
            CoreMLPrecision::Float32,
            CoreMLPrecision::Float16,
            CoreMLPrecision::Int8,
            CoreMLPrecision::Mixed,
        ];

        assert_eq!(precisions.len(), 4);
    }
}
