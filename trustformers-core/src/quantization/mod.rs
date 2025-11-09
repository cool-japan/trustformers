//! Quantization module for TrustformeRS
//!
//! This module provides various quantization techniques including:
//! - Standard INT8/INT4 quantization
//! - BitsAndBytes compatibility
//! - GPTQ and AWQ quantization
//! - Quantization-aware training (QAT)
//! - Learned quantization with trainable parameters
//! - SmoothQuant for W8A8 quantization
//! - Advanced GGML Q5/Q6 formats
//! - Activation quantization for runtime inference optimization
//! - Unified calibration toolkit for comprehensive quantization workflow management

mod activation;
mod base;
mod bitsandbytes;
mod calibration_toolkit;
mod ggml_advanced;
mod learned;
mod mixed_bit;
mod qat;
mod smoothquant;

// Re-export all items from base module
pub use base::{
    AWQQuantizer, BnBComputeType, BnBConfig, BnBQuantType, BnBQuantizer, BnBStorageType,
    FakeQuantize, GPTQQuantizer, Observer, QuantizationConfig, QuantizationScheme, QuantizedTensor,
    Quantizer,
};

// Re-export bitsandbytes specific items
pub use bitsandbytes::{
    dequantize_bitsandbytes, from_bitsandbytes_format, quantize_4bit, quantize_dynamic_tree,
    quantize_int8, to_bitsandbytes_format, BitsAndBytesConfig, QuantState,
};

// Re-export SmoothQuant items
pub use smoothquant::{
    MigrationAnalyzer, QuantizedTensor as SmoothQuantTensor, SmoothQuantConfig,
    SmoothQuantizedLinear, SmoothQuantizer,
};

// Re-export advanced GGML items
pub use ggml_advanced::{
    dequantize_q5_0, quantize_q5_0, quantize_q5_1, quantize_q6_k, AdvancedGGMLQuantizer, BlockQ5_0,
    BlockQ5_1, BlockQ6K, GGMLQuantType, QuantizedGGMLTensor,
};

// Re-export learned quantization items
pub use learned::{
    LearnedFakeQuantize, LearnedQuantConfig, LearnedQuantLayer, LearnedQuantOptimizer,
    LearnedQuantParams, LearnedQuantStats, LearnedQuantTrainer,
};

// Re-export mixed-bit quantization items
pub use mixed_bit::{
    AutoBitAllocationStrategy, LayerQuantConfig, MixedBitConfig, MixedBitQuantizedTensor,
    MixedBitQuantizer, QuantizedBlock, SensitivityConfig, SensitivityMetric,
};

// Re-export activation quantization items
pub use activation::{
    ActivationQuantConfig, ActivationQuantScheme, ActivationQuantizer, ActivationStats,
    LayerQuantConfig as ActivationLayerQuantConfig, QuantizedActivation,
};

// Re-export QAT items
pub use qat::{
    FakeQuantLayer, GradualSchedule, LayerSchedule, MovingAverageObserver, ObserverConfig,
    QATConfig, QATSchedule, QATState, QATStats, QATTrainer, QATUtils,
};

// Re-export calibration toolkit items
pub use calibration_toolkit::{
    CalibrationConfig, CalibrationDataset, CalibrationMetadata, CalibrationMethod,
    CalibrationParameter, CalibrationParameters, CalibrationRecommendation, CalibrationReport,
    CalibrationResult, CalibrationToolkit, CrossValidationConfig, CrossValidationResults,
    DatasetStatistics, DistributionAnalysis, DistributionType, DynamicRange, LayerQualityMetrics,
    MethodComparison, QualityMetrics, QualityThresholds, RecommendationType, TensorStatistics,
    TradeOffAnalysis,
};
