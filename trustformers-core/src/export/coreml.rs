// Core ML export functionality for iOS deployment
//! # Why no `.mlmodel` is written
//!
//! A Core ML model is an operation graph (`NeuralNetwork` / `MLProgram`) serialised
//! as an Apple protobuf. [`Model`] exposes parameters (via
//! [`Model::named_tensors`]) but not topology, so the layer list cannot be derived.
//!
//! Earlier revisions emitted a fixed 768-wide, 12-block transformer whose every
//! weight was `sin(i * 0.001)`, wrote it under the `.mlmodel` extension and
//! reported success. That artifact described a model that did not exist, so it is
//! no longer produced: [`CoreMLExporter::export`] returns a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation).
//!
//! Use the GGUF or GGML exporters to write the model's real parameters; they are
//! tensor containers and need no topology.

use super::{ExportConfig, ExportFormat, ModelExporter};
use crate::errors::unsupported_operation;
use crate::traits::Model;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Explanation attached to every refusal to write a Core ML model.
pub const COREML_UNSUPPORTED_REASON: &str =
    "a Core ML model is an operation graph serialised as an Apple protobuf; the \
     `Model` trait exposes parameters only (`named_tensors`), so TrustformeRS will \
     not emit a synthesized layer graph under a real model's name. Convert an ONNX \
     export with `coremltools` instead.";

/// Core ML model representation
#[derive(Debug, Clone)]
pub struct CoreMLModel {
    pub specification_version: u32,
    pub description: CoreMLModelDescription,
    pub neural_network: CoreMLNeuralNetwork,
    pub model_type: CoreMLModelType,
}

#[derive(Debug, Clone)]
pub enum CoreMLModelType {
    NeuralNetwork(CoreMLNeuralNetwork),
    Pipeline(CoreMLPipeline),
    MLProgram(CoreMLProgram),
}

#[derive(Debug, Clone)]
pub struct CoreMLPipeline {
    pub models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CoreMLProgram {
    pub functions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CoreMLModelDescription {
    pub input: Vec<CoreMLFeatureDescription>,
    pub output: Vec<CoreMLFeatureDescription>,
    pub predicted_feature_name: Option<String>,
    pub predicted_probabilities_name: Option<String>,
    pub training_input: Vec<CoreMLFeatureDescription>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CoreMLFeatureDescription {
    pub name: String,
    pub short_description: String,
    pub feature_type: CoreMLFeatureType,
}

#[derive(Debug, Clone)]
pub enum CoreMLFeatureType {
    MultiArray(CoreMLArrayFeatureType),
    String(CoreMLStringFeatureType),
    Int64(CoreMLInt64FeatureType),
    Double(CoreMLDoubleFeatureType),
    Dictionary(CoreMLDictionaryFeatureType),
    Sequence(Box<CoreMLFeatureType>),
}

#[derive(Debug, Clone)]
pub struct CoreMLArrayFeatureType {
    pub shape: Vec<i64>,
    pub data_type: CoreMLArrayDataType,
    pub default_optional_value: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy)]
pub enum CoreMLArrayDataType {
    Float32 = 65568,
    Float16 = 65552,
    Int32 = 131104,
}

#[derive(Debug, Clone)]
pub struct CoreMLStringFeatureType {
    pub default_value: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CoreMLInt64FeatureType {
    pub default_value: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct CoreMLDoubleFeatureType {
    pub default_value: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CoreMLDictionaryFeatureType {
    pub key_type: CoreMLDictionaryKeyType,
}

#[derive(Debug, Clone)]
pub enum CoreMLDictionaryKeyType {
    String,
    Int64,
}

/// Core ML Neural Network representation
#[derive(Debug, Clone)]
pub struct CoreMLNeuralNetwork {
    pub layers: Vec<CoreMLNeuralNetworkLayer>,
    pub preprocessing: Vec<CoreMLFeatureDescription>,
    pub array_inputs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CoreMLNeuralNetworkLayer {
    pub name: String,
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub layer_type: CoreMLLayerType,
}

#[derive(Debug, Clone)]
pub enum CoreMLLayerType {
    InnerProduct(CoreMLInnerProductLayer),
    Convolution(CoreMLConvolutionLayer),
    Activation(CoreMLActivationLayer),
    Pooling(CoreMLPoolingLayer),
    Normalization(CoreMLNormalizationLayer),
    Softmax(CoreMLSoftmaxLayer),
    LRN(CoreMLLRNLayer),
    Crop(CoreMLCropLayer),
    Padding(CoreMLPaddingLayer),
    Upsample(CoreMLUpsampleLayer),
    Unary(CoreMLUnaryLayer),
    Add(CoreMLAddLayer),
    Multiply(CoreMLMultiplyLayer),
    Average(CoreMLAverageLayer),
    Scale(CoreMLScaleLayer),
    Bias(CoreMLBiasLayer),
    Max(CoreMLMaxLayer),
    Min(CoreMLMinLayer),
    Dot(CoreMLDotLayer),
    Reduce(CoreMLReduceLayer),
    LoadConstant(CoreMLLoadConstantLayer),
    Reshape(CoreMLReshapeLayer),
    Flatten(CoreMLFlattenLayer),
    Permute(CoreMLPermuteLayer),
    Concat(CoreMLConcatLayer),
    Split(CoreMLSplitLayer),
    SequenceRepeat(CoreMLSequenceRepeatLayer),
    Reorganize(CoreMLReorganizeLayer),
    Slice(CoreMLSliceLayer),
    EmbeddingND(CoreMLEmbeddingNDLayer),
    BatchedMatMul(CoreMLBatchedMatMulLayer),
}

// Layer type definitions
#[derive(Debug, Clone)]
pub struct CoreMLInnerProductLayer {
    pub input_channels: u64,
    pub output_channels: u64,
    pub has_bias: bool,
    pub weights: CoreMLWeightParams,
    pub bias: Option<CoreMLWeightParams>,
}

#[derive(Debug, Clone)]
pub struct CoreMLConvolutionLayer {
    pub output_channels: u64,
    pub kernel_channels: u64,
    pub n_groups: u64,
    pub kernel_size: Vec<u64>,
    pub stride: Vec<u64>,
    pub dilation_factor: Vec<u64>,
    pub valid: CoreMLValidPadding,
    pub weights: CoreMLWeightParams,
    pub bias: Option<CoreMLWeightParams>,
    pub output_shape: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct CoreMLValidPadding {
    pub padding_amounts: CoreMLBorderAmounts,
}

#[derive(Debug, Clone)]
pub struct CoreMLBorderAmounts {
    pub border_amounts: Vec<CoreMLBorderAmount>,
}

#[derive(Debug, Clone)]
pub struct CoreMLBorderAmount {
    pub start_edge_size: u64,
    pub end_edge_size: u64,
}

#[derive(Debug, Clone)]
pub struct CoreMLActivationLayer {
    pub activation_type: CoreMLActivationType,
}

#[derive(Debug, Clone)]
pub enum CoreMLActivationType {
    ReLU,
    LeakyReLU { alpha: f32 },
    Tanh,
    Sigmoid,
    SoftPlus,
    SoftSign,
    ELU { alpha: f32 },
    PReLU { alpha: CoreMLWeightParams },
    ThresholdedReLU { alpha: f32 },
    Linear { alpha: f32, beta: f32 },
}

#[derive(Debug, Clone)]
pub struct CoreMLPoolingLayer {
    pub pooling_type: CoreMLPoolingType,
    pub kernel_size: Vec<u64>,
    pub stride: Vec<u64>,
    pub valid: CoreMLValidPadding,
    pub avg_pool_exclude_padding: bool,
    pub global_pooling: bool,
}

#[derive(Debug, Clone)]
pub enum CoreMLPoolingType {
    Max,
    Average,
    L2,
}

#[derive(Debug, Clone)]
pub struct CoreMLNormalizationLayer {
    pub normalization_type: CoreMLNormalizationType,
}

#[derive(Debug, Clone)]
pub enum CoreMLNormalizationType {
    LRN {
        alpha: f32,
        beta: f32,
        local_size: u64,
        k: f32,
    },
    BatchNorm {
        channels: u64,
        computed_mean: CoreMLWeightParams,
        computed_variance: CoreMLWeightParams,
        epsilon: f32,
    },
    InstanceNorm {
        channels: u64,
        epsilon: f32,
        gamma: Option<CoreMLWeightParams>,
        beta: Option<CoreMLWeightParams>,
    },
    LayerNorm {
        normalized_shape: Vec<u64>,
        eps: f32,
        gamma: Option<CoreMLWeightParams>,
        beta: Option<CoreMLWeightParams>,
    },
}

#[derive(Debug, Clone)]
pub struct CoreMLSoftmaxLayer {
    pub axis: i64,
}

#[derive(Debug, Clone)]
pub struct CoreMLLRNLayer {
    pub alpha: f32,
    pub beta: f32,
    pub local_size: u64,
    pub k: f32,
}

#[derive(Debug, Clone)]
pub struct CoreMLCropLayer {
    pub crop_amounts: CoreMLBorderAmounts,
    pub offset: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct CoreMLPaddingLayer {
    pub padding_type: CoreMLPaddingType,
}

#[derive(Debug, Clone)]
pub enum CoreMLPaddingType {
    Constant {
        value: f32,
        padding_amounts: CoreMLBorderAmounts,
    },
    Reflection {
        padding_amounts: CoreMLBorderAmounts,
    },
    Replication {
        padding_amounts: CoreMLBorderAmounts,
    },
}

#[derive(Debug, Clone)]
pub struct CoreMLUpsampleLayer {
    pub scaling_factor: Vec<u64>,
    pub mode: CoreMLUpsampleMode,
}

#[derive(Debug, Clone)]
pub enum CoreMLUpsampleMode {
    NN, // Nearest neighbor
    Bilinear,
}

#[derive(Debug, Clone)]
pub struct CoreMLUnaryLayer {
    pub unary_type: CoreMLUnaryType,
}

#[derive(Debug, Clone)]
pub enum CoreMLUnaryType {
    Sqrt,
    Rsqrt,
    Inverse,
    Power { alpha: f32 },
    Exp,
    Log,
    Abs,
    Threshold { alpha: f32 },
}

#[derive(Debug, Clone)]
pub struct CoreMLAddLayer {
    pub alpha: f32,
}

#[derive(Debug, Clone)]
pub struct CoreMLMultiplyLayer {
    pub alpha: f32,
}

#[derive(Debug, Clone)]
pub struct CoreMLAverageLayer;

#[derive(Debug, Clone)]
pub struct CoreMLScaleLayer {
    pub shape_scale: Vec<u64>,
    pub scale: CoreMLWeightParams,
    pub has_bias: bool,
    pub shape_bias: Vec<u64>,
    pub bias: Option<CoreMLWeightParams>,
}

#[derive(Debug, Clone)]
pub struct CoreMLBiasLayer {
    pub shape: Vec<u64>,
    pub bias: CoreMLWeightParams,
}

#[derive(Debug, Clone)]
pub struct CoreMLMaxLayer;

#[derive(Debug, Clone)]
pub struct CoreMLMinLayer;

#[derive(Debug, Clone)]
pub struct CoreMLDotLayer {
    pub cos_distance: bool,
}

#[derive(Debug, Clone)]
pub struct CoreMLReduceLayer {
    pub reduce_type: CoreMLReduceType,
    pub axis: i64,
    pub keep_dims: bool,
}

#[derive(Debug, Clone)]
pub enum CoreMLReduceType {
    Sum,
    Avg,
    Prod,
    LogSum,
    SumSquare,
    L1,
    L2,
    Max,
    Min,
    ArgMax,
}

#[derive(Debug, Clone)]
pub struct CoreMLLoadConstantLayer {
    pub shape: Vec<u64>,
    pub data: CoreMLWeightParams,
}

#[derive(Debug, Clone)]
pub struct CoreMLReshapeLayer {
    pub target_shape: Vec<i64>,
    pub mode: CoreMLReshapeMode,
}

#[derive(Debug, Clone)]
pub enum CoreMLReshapeMode {
    Channel,
    Width,
    Height,
}

#[derive(Debug, Clone)]
pub struct CoreMLFlattenLayer {
    pub mode: CoreMLFlattenMode,
}

#[derive(Debug, Clone)]
pub enum CoreMLFlattenMode {
    Channel,
    Width,
    Height,
}

#[derive(Debug, Clone)]
pub struct CoreMLPermuteLayer {
    pub axis: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct CoreMLConcatLayer {
    pub sequence_concat: bool,
}

#[derive(Debug, Clone)]
pub struct CoreMLSplitLayer {
    pub n_outputs: u64,
}

#[derive(Debug, Clone)]
pub struct CoreMLSequenceRepeatLayer {
    pub n_repetitions: u64,
}

#[derive(Debug, Clone)]
pub struct CoreMLReorganizeLayer {
    pub block_size: u64,
    pub mode: CoreMLReorganizeMode,
}

#[derive(Debug, Clone)]
pub enum CoreMLReorganizeMode {
    SpaceToDepth,
    DepthToSpace,
    PixelShuffle,
}

#[derive(Debug, Clone)]
pub struct CoreMLSliceLayer {
    pub start_index: i64,
    pub end_index: i64,
    pub stride: u64,
    pub axis: i64,
}

#[derive(Debug, Clone)]
pub struct CoreMLEmbeddingNDLayer {
    pub vocab_size: u64,
    pub embedding_size: u64,
    pub has_bias: bool,
    pub weights: CoreMLWeightParams,
    pub bias: Option<CoreMLWeightParams>,
}

#[derive(Debug, Clone)]
pub struct CoreMLBatchedMatMulLayer {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub weight_matrix_first_dimension: u64,
    pub weight_matrix_second_dimension: u64,
    pub has_bias: bool,
    pub weights: CoreMLWeightParams,
    pub bias: Option<CoreMLWeightParams>,
}

#[derive(Debug, Clone)]
pub struct CoreMLWeightParams {
    pub quantization: Option<CoreMLQuantizationParams>,
    pub float_value: Vec<f32>,
    pub float16_value: Vec<u16>,
    pub raw_value: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CoreMLQuantizationParams {
    pub number_of_bits: u64,
    pub linear_quantization: Option<CoreMLLinearQuantizationParams>,
    pub lookup_table_quantization: Option<CoreMLLookupTableQuantizationParams>,
}

#[derive(Debug, Clone)]
pub struct CoreMLLinearQuantizationParams {
    pub scale: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CoreMLLookupTableQuantizationParams {
    pub float_value: Vec<f32>,
}

/// Core ML exporter implementation
#[derive(Clone)]
pub struct CoreMLExporter {
    target_ios_version: String,
    optimization_enabled: bool,
}

impl Default for CoreMLExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreMLExporter {
    pub fn new() -> Self {
        Self {
            target_ios_version: "13.0".to_string(),
            optimization_enabled: true,
        }
    }

    pub fn with_target_ios_version(mut self, version: String) -> Self {
        self.target_ios_version = version;
        self
    }

    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.optimization_enabled = enabled;
        self
    }
}

impl ModelExporter for CoreMLExporter {
    /// Always fails with a structured `UnsupportedOperation` error.
    ///
    /// See the [module documentation](self) for why no `.mlmodel` is written.
    fn export<M: Model>(&self, model: &M, config: &ExportConfig) -> Result<()> {
        if config.format != ExportFormat::CoreML {
            return Err(anyhow!("CoreMLExporter only supports Core ML format"));
        }

        // Surface the "no weights at all" problem first: it is the caller's bug,
        // whereas the missing topology is a limitation of the `Model` trait.
        let _tensors = crate::export::collect_model_tensors(model)?;
        Err(unsupported_operation("Core ML model export", COREML_UNSUPPORTED_REASON).into())
    }

    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![ExportFormat::CoreML]
    }

    fn validate_model<M: Model>(&self, _model: &M, format: ExportFormat) -> Result<()> {
        if format != ExportFormat::CoreML {
            return Err(anyhow!("CoreMLExporter only supports Core ML format"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_exporter_creation() {
        let exporter = CoreMLExporter::new();
        assert_eq!(exporter.target_ios_version, "13.0");
        assert!(exporter.optimization_enabled);

        let exporter_custom =
            exporter.with_target_ios_version("14.0".to_string()).with_optimization(false);
        assert_eq!(exporter_custom.target_ios_version, "14.0");
        assert!(!exporter_custom.optimization_enabled);
    }

    #[test]
    fn test_coreml_array_data_types() {
        assert_eq!(CoreMLArrayDataType::Float32 as u32, 65568);
        assert_eq!(CoreMLArrayDataType::Float16 as u32, 65552);
        assert_eq!(CoreMLArrayDataType::Int32 as u32, 131104);
    }

    #[test]
    fn test_coreml_feature_types() {
        let array_feature = CoreMLFeatureType::MultiArray(CoreMLArrayFeatureType {
            shape: vec![1, 512],
            data_type: CoreMLArrayDataType::Float32,
            default_optional_value: None,
        });

        let string_feature = CoreMLFeatureType::String(CoreMLStringFeatureType {
            default_value: Some("default".to_string()),
        });

        match array_feature {
            CoreMLFeatureType::MultiArray(_) => {},
            _ => panic!(
                "Expected MultiArray feature type but got {:?}",
                array_feature
            ),
        }

        match string_feature {
            CoreMLFeatureType::String(_) => {},
            _ => panic!("Expected String feature type but got {:?}", string_feature),
        }
    }

    #[test]
    fn test_supported_formats() {
        let exporter = CoreMLExporter::new();
        let formats = exporter.supported_formats();
        assert_eq!(formats.len(), 1);
        assert_eq!(formats[0], ExportFormat::CoreML);
    }

    #[test]
    fn test_coreml_activation_types() {
        let relu = CoreMLActivationType::ReLU;
        let leaky_relu = CoreMLActivationType::LeakyReLU { alpha: 0.1 };
        let sigmoid = CoreMLActivationType::Sigmoid;

        match relu {
            CoreMLActivationType::ReLU => {},
            _ => panic!("Expected ReLU activation but got {:?}", relu),
        }

        match leaky_relu {
            CoreMLActivationType::LeakyReLU { alpha } => assert!((alpha - 0.1).abs() < 1e-6),
            _ => panic!("Expected LeakyReLU activation but got {:?}", leaky_relu),
        }

        match sigmoid {
            CoreMLActivationType::Sigmoid => {},
            _ => panic!("Expected Sigmoid activation but got {:?}", sigmoid),
        }
    }

    /// Regression test for the exporter that used to write a fixed 12-block
    /// transformer whose weights were `sin(i * 0.001)` for any model.
    #[test]
    fn export_refuses_to_write_a_synthesized_mlmodel() {
        let dir = std::env::temp_dir().join("trustformers_coreml_export_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let output = dir.join("model");

        let exporter = CoreMLExporter::new();
        let model = crate::export::test_support::TestModel::with_seed(5.0);
        let config = ExportConfig {
            format: ExportFormat::CoreML,
            output_path: output.to_string_lossy().to_string(),
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("must not fabricate a model");
        assert!(
            err.to_string().contains("Unsupported operation"),
            "expected UnsupportedOperation, got: {err}"
        );
        assert!(
            !output.with_extension("mlmodel").exists(),
            "no .mlmodel may be produced"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_reports_missing_weights_before_missing_topology() {
        let exporter = CoreMLExporter::new();
        let model = crate::export::test_support::TestModel::empty();
        let config = ExportConfig {
            format: ExportFormat::CoreML,
            ..Default::default()
        };

        let err = exporter.export(&model, &config).expect_err("no weights, no export");
        assert!(
            err.to_string().contains("named_tensors"),
            "expected the missing-weights diagnostic, got: {err}"
        );
    }
}
