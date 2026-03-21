//! Quantization configuration structures
//!
//! This module contains all configuration-related structures for the quantization system,
//! including main configurations, advanced settings, and performance parameters.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::*;

/// Main quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method
    pub quantization_type: QuantizationType,
    /// Weight precision
    pub weight_precision: QuantizationPrecision,
    /// Activation precision
    pub activation_precision: QuantizationPrecision,
    /// Quantize weights
    pub quantize_weights: bool,
    /// Quantize activations
    pub quantize_activations: bool,
    /// Calibration method
    pub calibration_method: CalibrationMethod,
    /// Quantization granularity
    pub granularity: QuantizationGranularity,
    /// Calibration dataset path
    pub calibration_dataset: Option<String>,
    /// Number of calibration samples
    pub calibration_samples: u32,
    /// Quantization range settings
    pub range_settings: RangeSettings,
    /// Layer-specific configurations
    pub layer_configs: HashMap<String, LayerQuantizationConfig>,
    /// Advanced settings
    pub advanced_settings: AdvancedQuantizationSettings,
    /// Performance optimization settings
    pub performance_settings: QuantizationPerformanceSettings,
}

/// Quantization range settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeSettings {
    /// Use symmetric quantization
    pub symmetric: bool,
    /// Use signed quantization
    pub signed: bool,
    /// Quantization range minimum
    pub range_min: Option<f64>,
    /// Quantization range maximum
    pub range_max: Option<f64>,
    /// Zero point
    pub zero_point: Option<i32>,
    /// Scale factor
    pub scale: Option<f64>,
    /// Clipping value
    pub clip_value: Option<f64>,
}

/// Layer-specific quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantizationConfig {
    /// Layer name pattern
    pub layer_pattern: String,
    /// Weight precision for this layer
    pub weight_precision: QuantizationPrecision,
    /// Activation precision for this layer
    pub activation_precision: QuantizationPrecision,
    /// Skip quantization for this layer
    pub skip_quantization: bool,
    /// Use different calibration method
    pub calibration_override: Option<CalibrationMethod>,
}

/// Advanced quantization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedQuantizationSettings {
    /// Enable outlier detection
    pub outlier_detection: bool,
    /// Outlier percentile threshold
    pub outlier_percentile: f64,
    /// Enable smoothing
    pub enable_smoothing: bool,
    /// Smoothing alpha parameter
    pub smoothing_alpha: f64,
    /// Enable channel shuffling
    pub channel_shuffling: bool,
    /// Use knowledge distillation
    pub knowledge_distillation: bool,
    /// Temperature for knowledge distillation
    pub distillation_temperature: f64,
    /// Gradient scaling for mixed precision
    pub gradient_scaling: bool,
    /// Loss scaling factor
    pub loss_scale_factor: f64,
    /// Mixed-bit quantization configuration
    pub mixed_bit_config: Option<MixedBitConfig>,
    /// Learned quantization parameters
    pub learned_quantization: Option<LearnedQuantizationParams>,
    /// Enhanced QAT configuration
    pub enhanced_qat_config: Option<EnhancedQATConfig>,
    /// BitsAndBytes compatibility configuration
    pub bitsandbytes_config: Option<BitsAndBytesConfig>,
}

/// Performance optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationPerformanceSettings {
    /// Enable kernel fusion
    pub kernel_fusion: bool,
    /// Use optimized kernels
    pub optimized_kernels: bool,
    /// Enable parallel quantization
    pub parallel_quantization: bool,
    /// Number of quantization threads
    pub num_threads: u32,
    /// Memory optimization level
    pub memory_optimization: u8,
    /// Cache quantized weights
    pub cache_quantized_weights: bool,
}

/// Mixed-bit quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MixedBitConfig {
    /// Enable mixed-bit quantization
    pub enabled: bool,
    /// Bit allocation strategy
    pub allocation_strategy: BitAllocationStrategy,
    /// Available bit widths
    pub available_bits: Vec<u8>,
    /// Sensitivity threshold
    pub sensitivity_threshold: f64,
    /// Performance weight in allocation
    pub performance_weight: f64,
    /// Memory weight in allocation
    pub memory_weight: f64,
    /// Accuracy weight in allocation
    pub accuracy_weight: f64,
    /// Layer-specific bit assignments
    pub layer_bit_assignments: HashMap<String, u8>,
    /// Channel-specific bit assignments
    pub channel_bit_assignments: HashMap<String, Vec<u8>>,
}

/// Learned quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedQuantizationParams {
    /// Enable learned quantization
    pub enabled: bool,
    /// Learning rate for quantization parameters
    pub learning_rate: f64,
    /// Number of training epochs
    pub num_epochs: u32,
    /// Batch size for learning
    pub batch_size: u32,
    /// Temperature annealing settings
    pub temperature_annealing: bool,
    /// Initial temperature
    pub initial_temperature: f64,
    /// Final temperature
    pub final_temperature: f64,
    /// Regularization strength
    pub regularization_strength: f64,
    /// Adaptive learning rate parameters
    pub adaptive_lr_params: Option<AdaptiveLearningRateParams>,
}

/// Adaptive learning rate parameters for learned quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningRateParams {
    /// Enable adaptive learning rate
    pub enabled: bool,
    /// Learning rate decay factor
    pub decay_factor: f64,
    /// Patience for learning rate reduction
    pub patience: u32,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Learning rate schedule type
    pub schedule_type: String,
}

/// Enhanced Quantization-Aware Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQATConfig {
    /// Enable enhanced QAT
    pub enabled: bool,
    /// Fake quantization configuration
    pub fake_quant_config: FakeQuantConfig,
    /// QAT training schedule
    pub qat_schedule: QATSchedule,
    /// Knowledge distillation configuration
    pub distillation_config: Option<QATDistillationConfig>,
    /// Progressive quantization schedule
    pub progressive_schedule: Option<ProgressiveQuantSchedule>,
    /// Noise injection configuration
    pub noise_injection: Option<NoiseInjectionConfig>,
    /// Straight-Through Estimator configuration
    pub ste_config: STEConfig,
}

/// QAT training schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QATSchedule {
    /// Total training steps
    pub total_steps: u32,
    /// Warmup steps
    pub warmup_steps: u32,
    /// Activation quantization schedule
    pub activation_schedule: ActivationSchedule,
    /// Weight quantization schedule
    pub weight_schedule: WeightSchedule,
    /// Schedule-specific parameters
    pub schedule_params: HashMap<String, f64>,
}

/// Fake quantization configuration for QAT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FakeQuantConfig {
    /// Observer type for fake quantization
    pub observer_type: ObserverType,
    /// Quantization scheme
    pub quant_scheme: QuantizationScheme,
    /// Number of observer update steps
    pub observer_update_steps: u32,
    /// Observer momentum
    pub observer_momentum: f64,
    /// Quantization delay (steps before starting quantization)
    pub quant_delay: u32,
    /// Range calibration method
    pub range_calibration: RangeCalibrationMethod,
}

/// BitsAndBytes compatibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitsAndBytesConfig {
    /// Enable BitsAndBytes quantization
    pub enabled: bool,
    /// Quantization type
    pub quant_type: BnBQuantType,
    /// 4-bit configuration
    pub bnb_4bit_config: Option<BnB4BitConfig>,
    /// 8-bit configuration
    pub bnb_8bit_config: Option<BnB8BitConfig>,
    /// Use double quantization
    pub use_double_quant: bool,
    /// Compute dtype for operations
    pub compute_dtype: String,
    /// Quantization statistics computation dtype
    pub quant_storage_dtype: String,
}

/// BitsAndBytes 4-bit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BnB4BitConfig {
    /// Use 4-bit quantization
    pub use_4bit: bool,
    /// 4-bit compute dtype
    pub bnb_4bit_compute_dtype: String,
    /// Use nested quantization
    pub use_nested_quant: bool,
    /// Block size for quantization
    pub bnb_4bit_quant_type: String,
}

/// BitsAndBytes 8-bit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BnB8BitConfig {
    /// Use 8-bit quantization
    pub use_8bit: bool,
    /// Threshold for outlier detection
    pub llm_int8_threshold: f64,
    /// Skip modules containing these strings
    pub llm_int8_skip_modules: Vec<String>,
    /// Enable safe int8 matrix multiplication
    pub llm_int8_enable_fp32_cpu_offload: bool,
}

/// QAT knowledge distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QATDistillationConfig {
    /// Enable knowledge distillation
    pub enabled: bool,
    /// Teacher model path
    pub teacher_model_path: String,
    /// Distillation temperature
    pub temperature: f64,
    /// Distillation loss weight
    pub distillation_weight: f64,
    /// Hard target loss weight
    pub hard_target_weight: f64,
    /// Feature distillation layers
    pub feature_distillation_layers: Vec<String>,
    /// Attention distillation weight
    pub attention_distillation_weight: f64,
}

/// Progressive quantization schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveQuantSchedule {
    /// Enable progressive quantization
    pub enabled: bool,
    /// Schedule type
    pub schedule_type: ProgressiveScheduleType,
    /// Number of progressive steps
    pub num_steps: u32,
    /// Step duration (training steps per progressive step)
    pub step_duration: u32,
    /// Initial bit width
    pub initial_bits: u8,
    /// Final bit width
    pub final_bits: u8,
}

/// Noise injection configuration for quantization robustness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseInjectionConfig {
    /// Enable noise injection
    pub enabled: bool,
    /// Noise type
    pub noise_type: NoiseType,
    /// Noise schedule
    pub noise_schedule: NoiseSchedule,
    /// Initial noise magnitude
    pub initial_magnitude: f64,
    /// Final noise magnitude
    pub final_magnitude: f64,
    /// Layers to inject noise into
    pub target_layers: Vec<String>,
}

/// Noise schedule for quantization training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSchedule {
    /// Schedule type (linear, exponential, etc.)
    pub schedule_type: String,
    /// Schedule parameters
    pub schedule_params: HashMap<String, f64>,
    /// Annealing steps
    pub annealing_steps: u32,
}

/// Straight-Through Estimator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STEConfig {
    /// STE type
    pub ste_type: STEType,
    /// Temperature for soft quantization
    pub temperature: f64,
    /// Gradient clipping threshold
    pub gradient_clipping: Option<f64>,
    /// Learnable temperature (for learnable STE)
    pub learnable_temperature: bool,
    /// Temperature learning rate
    pub temperature_lr: f64,
}

/// Quantization parameters for tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f64,
    /// Zero point
    pub zero_point: i32,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Bit width
    pub bit_width: u8,
    /// Signed/unsigned
    pub signed: bool,
    /// Per-channel parameters (if applicable)
    pub per_channel_scales: Option<Vec<f64>>,
    /// Per-channel zero points (if applicable)
    pub per_channel_zero_points: Option<Vec<i32>>,
}

/// C API configuration structure for external interfaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustformersQuantizationConfig {
    /// Configuration version
    pub version: u32,
    /// JSON-serialized QuantizationConfig
    pub config_json: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

// Default implementations

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::default(),
            weight_precision: QuantizationPrecision::INT8,
            activation_precision: QuantizationPrecision::INT8,
            quantize_weights: true,
            quantize_activations: false,
            calibration_method: CalibrationMethod::default(),
            granularity: QuantizationGranularity::default(),
            calibration_dataset: None,
            calibration_samples: 100,
            range_settings: RangeSettings::default(),
            layer_configs: HashMap::new(),
            advanced_settings: AdvancedQuantizationSettings::default(),
            performance_settings: QuantizationPerformanceSettings::default(),
        }
    }
}

impl Default for RangeSettings {
    fn default() -> Self {
        Self {
            symmetric: true,
            signed: true,
            range_min: None,
            range_max: None,
            zero_point: None,
            scale: None,
            clip_value: None,
        }
    }
}

impl Default for AdvancedQuantizationSettings {
    fn default() -> Self {
        Self {
            outlier_detection: false,
            outlier_percentile: 99.9,
            enable_smoothing: false,
            smoothing_alpha: 0.1,
            channel_shuffling: false,
            knowledge_distillation: false,
            distillation_temperature: 3.0,
            gradient_scaling: false,
            loss_scale_factor: 1.0,
            mixed_bit_config: None,
            learned_quantization: None,
            enhanced_qat_config: None,
            bitsandbytes_config: None,
        }
    }
}

impl Default for QuantizationPerformanceSettings {
    fn default() -> Self {
        Self {
            kernel_fusion: true,
            optimized_kernels: true,
            parallel_quantization: true,
            num_threads: num_cpus::get() as u32,
            memory_optimization: 1,
            cache_quantized_weights: true,
        }
    }
}
