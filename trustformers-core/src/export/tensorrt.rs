//! TensorRT export surface.
//!
//! # Why there is no TensorRT engine writer here
//!
//! A TensorRT `.plan` file is a *serialised engine*: the output of NVIDIA's
//! closed-source builder (`nvinfer`), which performs kernel auto-tuning against the
//! exact GPU architecture, driver and TensorRT version present on the machine that
//! builds it. The byte layout is deliberately unspecified, version-locked, and can
//! only be produced by linking against NVIDIA's C++ runtime.
//!
//! That is impossible to produce from pure Rust, and linking the C++ runtime is
//! barred by this workspace's pure-Rust policy. Rather than write a look-alike file,
//! [`TensorRTExporter::export`] returns a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation)
//! error. The configuration and network description types below remain public so
//! that callers can describe an intended engine and hand it to an external
//! TensorRT toolchain.

use super::{ExportConfig, ExportFormat, ModelExporter};
use crate::errors::unsupported_operation;
use crate::traits::Model;
use anyhow::{anyhow, Result};

/// Explanation attached to every refusal to write a TensorRT engine.
pub const TENSORRT_UNSUPPORTED_REASON: &str =
    "a TensorRT `.plan` file is a serialised engine produced by NVIDIA's closed-source \
     builder, which auto-tunes kernels for the exact GPU, driver and TensorRT version \
     of the building machine; it cannot be produced in pure Rust and this crate will \
     not write a look-alike file. Build the engine with `trtexec`/the TensorRT Python \
     or C++ API from an ONNX model instead.";

/// TensorRT engine configuration
#[derive(Debug, Clone)]
pub struct TensorRTConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub workspace_size: usize, // in MB
    pub fp16_enabled: bool,
    pub int8_enabled: bool,
    pub dynamic_shapes: bool,
    pub optimization_level: u8, // 0-5
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequence_length: 2048,
            workspace_size: 1024, // 1GB
            fp16_enabled: true,
            int8_enabled: false,
            dynamic_shapes: true,
            optimization_level: 3,
        }
    }
}

/// TensorRT network representation
#[derive(Debug)]
pub struct TensorRTNetwork {
    pub layers: Vec<TensorRTLayer>,
    pub inputs: Vec<TensorRTTensor>,
    pub outputs: Vec<TensorRTTensor>,
}

#[derive(Debug)]
pub struct TensorRTLayer {
    pub layer_type: TensorRTLayerType,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub parameters: Vec<u8>, // Serialized parameters
}

#[derive(Debug, Clone)]
pub enum TensorRTLayerType {
    Convolution,
    FullyConnected,
    Activation,
    Pooling,
    ElementWise,
    Softmax,
    Concatenation,
    MatrixMultiply,
    Gather,
    Scatter,
    LayerNorm,
    MultiHeadAttention,
    Embedding,
    PositionalEncoding,
    RNN,
    Plugin(String), // Custom plugin name
}

#[derive(Debug)]
pub struct TensorRTTensor {
    pub name: String,
    pub dimensions: Vec<i32>, // -1 for dynamic dimensions
    pub data_type: TensorRTDataType,
}

#[derive(Debug, Clone, Copy)]
pub enum TensorRTDataType {
    Float32,
    Float16,
    Int8,
    Int32,
    Bool,
}

/// TensorRT exporter implementation
#[derive(Clone)]
pub struct TensorRTExporter {
    config: TensorRTConfig,
}

impl Default for TensorRTExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorRTExporter {
    pub fn new() -> Self {
        Self {
            config: TensorRTConfig::default(),
        }
    }

    pub fn with_config(mut self, config: TensorRTConfig) -> Self {
        self.config = config;
        self
    }

    /// The engine-builder configuration this exporter was constructed with.
    pub fn config(&self) -> &TensorRTConfig {
        &self.config
    }
}

impl ModelExporter for TensorRTExporter {
    /// Always fails with a structured `UnsupportedOperation` error.
    ///
    /// See the [module documentation](self) for why no `.plan` file is written.
    /// Earlier revisions of this exporter wrote an ASCII description under the
    /// `.plan` extension and reported success; that file was never loadable by
    /// `nvinfer`, so it is no longer produced.
    fn export<M: Model>(&self, _model: &M, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::TensorRT {
            return Err(anyhow!("TensorRTExporter only supports TensorRT format"));
        }

        Err(unsupported_operation(
            "TensorRT engine serialization",
            format!(
                "pure-Rust TrustformeRS build: {}",
                TENSORRT_UNSUPPORTED_REASON
            ),
        )
        .into())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::TensorRT]
    }

    fn validate_model<M: Model>(&self, _model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::TensorRT {
            return Err(anyhow!("TensorRTExporter only supports TensorRT format"));
        }

        // Additional validation could check for TensorRT compatibility
        // - Supported layer types
        // - Dynamic shape constraints
        // - Memory requirements

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::test_support::TestModel;
    use crate::export::ExportPrecision;

    /// Regression test for the exporter that used to write an ASCII description
    /// under the `.plan` extension and return `Ok(())`.
    #[test]
    fn export_refuses_to_write_a_plan_file() {
        let dir = std::env::temp_dir().join("trustformers_tensorrt_export_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let output = dir.join("engine");

        let exporter = TensorRTExporter::new();
        let model = TestModel::with_seed(1.0);
        let config = ExportConfig {
            format: ExportFormat::TensorRT,
            output_path: output.to_string_lossy().to_string(),
            precision: ExportPrecision::FP16,
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("must not fabricate an engine");
        let message = err.to_string();
        assert!(
            message.contains("Unsupported operation"),
            "expected a structured UnsupportedOperation error, got: {message}"
        );

        assert!(
            !output.with_extension("plan").exists(),
            "no .plan file may be produced"
        );
        assert!(
            !dir.join("engine_tensorrt.json").exists(),
            "no side-car description may be produced"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_rejects_non_tensorrt_formats() {
        let exporter = TensorRTExporter::new();
        let model = TestModel::with_seed(0.0);
        let config = ExportConfig {
            format: ExportFormat::GGUF,
            ..Default::default()
        };
        assert!(exporter.export(&model, &config).is_err());
    }

    #[test]
    fn test_tensorrt_exporter_creation() {
        let exporter = TensorRTExporter::new();
        assert_eq!(exporter.config.max_batch_size, 32);
        assert_eq!(exporter.config.max_sequence_length, 2048);
        assert!(exporter.config.fp16_enabled);
        assert!(!exporter.config.int8_enabled);
    }

    #[test]
    fn test_tensorrt_config_custom() {
        let config = TensorRTConfig {
            max_batch_size: 64,
            max_sequence_length: 4096,
            workspace_size: 2048,
            fp16_enabled: false,
            int8_enabled: true,
            dynamic_shapes: false,
            optimization_level: 5,
        };

        let exporter = TensorRTExporter::new().with_config(config);
        assert_eq!(exporter.config.max_batch_size, 64);
        assert_eq!(exporter.config.max_sequence_length, 4096);
        assert_eq!(exporter.config.workspace_size, 2048);
        assert!(!exporter.config.fp16_enabled);
        assert!(exporter.config.int8_enabled);
        assert!(!exporter.config.dynamic_shapes);
        assert_eq!(exporter.config.optimization_level, 5);
    }

    #[test]
    fn test_tensorrt_data_types() {
        let float32 = TensorRTDataType::Float32;
        let float16 = TensorRTDataType::Float16;
        let int8 = TensorRTDataType::Int8;
        let int32 = TensorRTDataType::Int32;
        let bool_type = TensorRTDataType::Bool;

        // Just test that all types exist and can be created
        assert!(matches!(float32, TensorRTDataType::Float32));
        assert!(matches!(float16, TensorRTDataType::Float16));
        assert!(matches!(int8, TensorRTDataType::Int8));
        assert!(matches!(int32, TensorRTDataType::Int32));
        assert!(matches!(bool_type, TensorRTDataType::Bool));
    }

    #[test]
    fn test_tensorrt_layer_types() {
        let layer_types = [
            TensorRTLayerType::Convolution,
            TensorRTLayerType::FullyConnected,
            TensorRTLayerType::Activation,
            TensorRTLayerType::MultiHeadAttention,
            TensorRTLayerType::LayerNorm,
            TensorRTLayerType::Plugin("custom_plugin".to_string()),
        ];

        assert_eq!(layer_types.len(), 6);

        match &layer_types[5] {
            TensorRTLayerType::Plugin(name) => assert_eq!(name, "custom_plugin"),
            _ => panic!("Expected Plugin layer type but got {:?}", &layer_types[5]),
        }
    }

    #[test]
    fn test_supported_formats() {
        let exporter = TensorRTExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats.len(), 1);
        assert_eq!(formats[0], ExportFormat::TensorRT);
    }

    #[test]
    fn test_tensorrt_tensor_creation() {
        let tensor = TensorRTTensor {
            name: "test_tensor".to_string(),
            dimensions: vec![-1, 512, 768],
            data_type: TensorRTDataType::Float32,
        };

        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.dimensions, vec![-1, 512, 768]);
        assert!(matches!(tensor.data_type, TensorRTDataType::Float32));
    }
}
