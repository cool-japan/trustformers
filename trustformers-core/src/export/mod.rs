// Model export functionality for various formats
pub mod async_export;
pub mod coreml;
pub mod factory;
pub mod ggml;
pub mod gguf;
pub mod gguf_enhanced;
pub mod nnef;
pub mod onnx;
pub mod onnx_runtime;
pub mod openvino;
pub mod optimization;
pub mod tensorrt;
pub mod tvm;

pub use async_export::{
    export_model_async, AsyncExportHandle, AsyncExportManager, ExportProgress, ExportStep,
};
pub use coreml::*;
pub use factory::{
    ExportConstraints, ExportResult, ExporterFactory, ExporterProvider, ExporterRequirements,
    TargetPlatform, ValidationResult,
};
pub use ggml::*;
pub use gguf::*;
pub use gguf_enhanced::{GGUFConverter, GGUFExporter as EnhancedGGUFExporter, GGUFTensorType};
pub use nnef::*;
pub use onnx::*;
pub use onnx_runtime::*;
pub use openvino::*;
pub use optimization::{
    OptimizationConfig, OptimizationImpact, OptimizationPass, OptimizationPipeline,
    OptimizationStats, PipelineStats, TargetHardware,
};
pub use tensorrt::*;
pub use tvm::*;

use crate::traits::Model;
use anyhow::Result;

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    ONNX,
    GGML,
    GGUF,
    NNEF,
    OpenVINO,
    TensorRT,
    TVM,
    CoreML,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub output_path: String,
    pub optimize: bool,
    pub precision: ExportPrecision,
    pub batch_size: Option<usize>,
    pub sequence_length: Option<usize>,
    pub opset_version: Option<i64>, // For ONNX
    pub quantization: Option<ExportQuantization>,
    pub input_shape: Option<Vec<usize>>,
    pub task_type: Option<String>,
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
}

#[derive(Debug, Clone)]
pub struct ExportQuantization {
    pub bits: u8,
    pub group_size: Option<usize>,
    pub calibration_data: Option<Vec<String>>, // Sample texts for calibration
}

/// Main export trait
pub trait ModelExporter {
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()>;

    fn supported_formats(&self) -> Vec<ExportFormat>;

    fn validate_model<M: Model>(&self, model: &M, format: ExportFormat) -> Result<()>;
}

/// Universal model exporter
#[derive(Clone)]
pub struct UniversalExporter;

impl Default for UniversalExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl UniversalExporter {
    pub fn new() -> Self {
        Self
    }

    pub fn export_model<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        self.validate_model(model, config.format)?;

        match config.format {
            ExportFormat::ONNX => {
                let exporter = ONNXExporter::new();
                exporter.export(model, config)
            },
            ExportFormat::GGML => {
                let exporter = GGMLExporter::new();
                exporter.export(model, config)
            },
            ExportFormat::GGUF => {
                let exporter = GGUFExporter::new();
                exporter.export(model, config)
            },
            ExportFormat::NNEF => {
                let exporter = NNEFExporter::new();
                exporter.export(model, config)
            },
            ExportFormat::OpenVINO => {
                let exporter = OpenVINOExporter::new();
                exporter.export(model, config)
            },
            ExportFormat::TensorRT => {
                let exporter = TensorRTExporter::new();
                exporter.export(model, config)
            },
            ExportFormat::TVM => {
                let exporter = TVMExporter::new();
                exporter.export(model, config)
            },
            ExportFormat::CoreML => {
                let exporter = CoreMLExporter::new();
                exporter.export(model, config)
            },
        }
    }
}

/// Concrete enum holding all exporter types for dyn compatibility
#[derive(Clone)]
pub enum ConcreteExporter {
    ONNX(ONNXExporter),
    GGML(GGMLExporter),
    GGUF(GGUFExporter),
    GGUFEnhanced(EnhancedGGUFExporter),
    NNEF(NNEFExporter),
    OpenVINO(OpenVINOExporter),
    TensorRT(TensorRTExporter),
    TVM(TVMExporter),
    CoreML(CoreMLExporter),
    Universal(UniversalExporter),
}

impl ModelExporter for ConcreteExporter {
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        match self {
            ConcreteExporter::ONNX(exporter) => exporter.export(model, config),
            ConcreteExporter::GGML(exporter) => exporter.export(model, config),
            ConcreteExporter::GGUF(exporter) => exporter.export(model, config),
            ConcreteExporter::GGUFEnhanced(exporter) => exporter.export(model, config),
            ConcreteExporter::NNEF(exporter) => exporter.export(model, config),
            ConcreteExporter::OpenVINO(exporter) => exporter.export(model, config),
            ConcreteExporter::TensorRT(exporter) => exporter.export(model, config),
            ConcreteExporter::TVM(exporter) => exporter.export(model, config),
            ConcreteExporter::CoreML(exporter) => exporter.export(model, config),
            ConcreteExporter::Universal(exporter) => exporter.export(model, config),
        }
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        match self {
            ConcreteExporter::ONNX(exporter) => exporter.supported_formats(),
            ConcreteExporter::GGML(exporter) => exporter.supported_formats(),
            ConcreteExporter::GGUF(exporter) => exporter.supported_formats(),
            ConcreteExporter::GGUFEnhanced(exporter) => exporter.supported_formats(),
            ConcreteExporter::NNEF(exporter) => exporter.supported_formats(),
            ConcreteExporter::OpenVINO(exporter) => exporter.supported_formats(),
            ConcreteExporter::TensorRT(exporter) => exporter.supported_formats(),
            ConcreteExporter::TVM(exporter) => exporter.supported_formats(),
            ConcreteExporter::CoreML(exporter) => exporter.supported_formats(),
            ConcreteExporter::Universal(exporter) => exporter.supported_formats(),
        }
    }

    fn validate_model<M: Model>(&self, model: &M, format: ExportFormat) -> Result<()> {
        match self {
            ConcreteExporter::ONNX(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::GGML(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::GGUF(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::GGUFEnhanced(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::NNEF(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::OpenVINO(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::TensorRT(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::TVM(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::CoreML(exporter) => exporter.validate_model(model, format),
            ConcreteExporter::Universal(exporter) => exporter.validate_model(model, format),
        }
    }
}

impl ModelExporter for UniversalExporter {
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        self.export_model(model, config)
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![
            ExportFormat::ONNX,
            ExportFormat::GGML,
            ExportFormat::GGUF,
            ExportFormat::NNEF,
            ExportFormat::OpenVINO,
            ExportFormat::TensorRT,
            ExportFormat::TVM,
            ExportFormat::CoreML,
        ]
    }

    fn validate_model<M: Model>(&self, _model: &M, _format: ExportFormat) -> Result<()> {
        // Basic validation - can be extended per format
        Ok(())
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::ONNX,
            output_path: "model".to_string(),
            optimize: true,
            precision: ExportPrecision::FP32,
            batch_size: Some(1),
            sequence_length: Some(512),
            opset_version: Some(14),
            quantization: None,
            input_shape: None,
            task_type: None,
            vocab_size: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::ONNX);
        assert_eq!(config.output_path, "model");
        assert!(config.optimize);
        assert!(matches!(config.precision, ExportPrecision::FP32));
    }

    #[test]
    fn test_universal_exporter_creation() {
        let exporter = UniversalExporter::new();
        let formats = exporter.supported_formats();
        assert!(formats.contains(&ExportFormat::ONNX));
        assert!(formats.contains(&ExportFormat::GGML));
        assert!(formats.contains(&ExportFormat::GGUF));
    }

    #[test]
    fn test_export_precision_variants() {
        let precisions = [
            ExportPrecision::FP32,
            ExportPrecision::FP16,
            ExportPrecision::INT8,
            ExportPrecision::INT4,
        ];

        for precision in precisions.iter() {
            let config = ExportConfig {
                precision: *precision,
                ..Default::default()
            };
            // Just test that we can create configs with different precisions
            assert!(matches!(
                config.precision,
                ExportPrecision::FP32
                    | ExportPrecision::FP16
                    | ExportPrecision::INT8
                    | ExportPrecision::INT4
            ));
        }
    }
}
