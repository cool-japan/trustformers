//! Quantization type definitions and enums
//!
//! This module contains all the core type definitions, enums, and basic data structures
//! used throughout the quantization system.

use serde::{Deserialize, Serialize};

/// Quantization method types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum QuantizationType {
    /// No quantization
    None = 0,
    /// Static INT8 quantization
    INT8 = 1,
    /// Static INT4 quantization
    INT4 = 2,
    /// Dynamic quantization
    Dynamic = 3,
    /// Mixed precision quantization
    MixedPrecision = 4,
    /// Quantization-aware training
    QAT = 5,
    /// GPTQ (GPT Quantization)
    GPTQ = 6,
    /// AWQ (Activation-aware Weight Quantization)
    AWQ = 7,
    /// SmoothQuant
    SmoothQuant = 8,
    /// GGML quantization
    GGML = 9,
    /// Custom quantization
    Custom = 99,
}

/// Quantization precision types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum QuantizationPrecision {
    /// 32-bit floating point
    FP32 = 32,
    /// 16-bit floating point
    FP16 = 16,
    /// 16-bit brain floating point
    BF16 = 15,
    /// 8-bit integer
    INT8 = 8,
    /// 4-bit integer
    INT4 = 4,
    /// 2-bit integer
    INT2 = 2,
    /// 1-bit integer (binary)
    INT1 = 1,
}

/// Quantization calibration method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum CalibrationMethod {
    /// Minimum-Maximum calibration
    MinMax = 0,
    /// Entropy/KL-divergence calibration
    Entropy = 1,
    /// Percentile calibration
    Percentile = 2,
    /// Mean-Standard deviation calibration
    MeanStd = 3,
    /// Histogram-based calibration
    Histogram = 4,
}

/// Quantization granularity
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum QuantizationGranularity {
    /// Per-tensor quantization
    PerTensor = 0,
    /// Per-channel quantization
    PerChannel = 1,
    /// Per-group quantization
    PerGroup = 2,
    /// Per-token quantization
    PerToken = 3,
}

/// Bit allocation strategy for mixed-bit quantization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum BitAllocationStrategy {
    /// Uniform bit allocation
    Uniform = 0,
    /// Sensitivity-based allocation
    SensitivityBased = 1,
    /// Hessian-based allocation
    HessianBased = 2,
    /// Fisher information-based allocation
    FisherBased = 3,
    /// Gradient-based allocation
    GradientBased = 4,
    /// Layer-wise adaptive allocation
    LayerAdaptive = 5,
    /// Channel-wise adaptive allocation
    ChannelAdaptive = 6,
}

/// Activation schedule types for QAT
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum ActivationSchedule {
    /// No schedule (constant)
    None = 0,
    /// Linear schedule
    Linear = 1,
    /// Exponential schedule
    Exponential = 2,
    /// Cosine schedule
    Cosine = 3,
    /// Step schedule
    Step = 4,
}

/// Weight schedule types for QAT
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum WeightSchedule {
    /// No schedule (constant)
    None = 0,
    /// Linear schedule
    Linear = 1,
    /// Exponential schedule
    Exponential = 2,
    /// Polynomial schedule
    Polynomial = 3,
    /// Custom schedule
    Custom = 4,
}

/// Observer types for QAT
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum ObserverType {
    /// MinMax observer
    MinMax = 0,
    /// MovingAverage observer
    MovingAverage = 1,
    /// Histogram observer
    Histogram = 2,
    /// Percentile observer
    Percentile = 3,
    /// Entropy observer
    Entropy = 4,
}

/// Quantization scheme types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum QuantizationScheme {
    /// Symmetric quantization
    Symmetric = 0,
    /// Asymmetric quantization
    Asymmetric = 1,
    /// Channel-wise symmetric
    ChannelSymmetric = 2,
    /// Channel-wise asymmetric
    ChannelAsymmetric = 3,
}

/// Range calibration methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum RangeCalibrationMethod {
    /// Global range calibration
    Global = 0,
    /// Layer-wise range calibration
    LayerWise = 1,
    /// Channel-wise range calibration
    ChannelWise = 2,
    /// Block-wise range calibration
    BlockWise = 3,
}

/// BitsAndBytes quantization types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum BnBQuantType {
    /// FP4 quantization
    FP4 = 0,
    /// NF4 quantization
    NF4 = 1,
    /// INT8 quantization
    INT8 = 2,
}

/// Progressive schedule types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum ProgressiveScheduleType {
    /// Layer-by-layer progression
    LayerByLayer = 0,
    /// Bit-by-bit progression
    BitByBit = 1,
    /// Channel-by-channel progression
    ChannelByChannel = 2,
    /// Group-by-group progression
    GroupByGroup = 3,
}

/// Noise types for quantization noise injection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum NoiseType {
    /// Gaussian noise
    Gaussian = 0,
    /// Uniform noise
    Uniform = 1,
    /// Laplacian noise
    Laplacian = 2,
    /// Bernoulli noise
    Bernoulli = 3,
}

/// Straight-Through Estimator types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum STEType {
    /// Standard STE
    Standard = 0,
    /// Improved STE
    Improved = 1,
    /// Learnable STE
    Learnable = 2,
    /// Adaptive STE
    Adaptive = 3,
}

/// Layer types for quantization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum LayerType {
    /// Linear/Dense layer
    Linear = 0,
    /// Convolutional layer
    Conv = 1,
    /// Attention layer
    Attention = 2,
    /// Normalization layer
    Norm = 3,
    /// Activation layer
    Activation = 4,
    /// Embedding layer
    Embedding = 5,
    /// Output layer
    Output = 6,
    /// Unknown layer type
    Unknown = 99,
}

impl Default for QuantizationType {
    fn default() -> Self {
        Self::None
    }
}

impl Default for QuantizationPrecision {
    fn default() -> Self {
        Self::FP32
    }
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        Self::MinMax
    }
}

impl Default for QuantizationGranularity {
    fn default() -> Self {
        Self::PerTensor
    }
}

impl Default for BitAllocationStrategy {
    fn default() -> Self {
        Self::Uniform
    }
}

impl Default for ObserverType {
    fn default() -> Self {
        Self::MinMax
    }
}

impl Default for QuantizationScheme {
    fn default() -> Self {
        Self::Symmetric
    }
}

impl Default for LayerType {
    fn default() -> Self {
        Self::Unknown
    }
}
