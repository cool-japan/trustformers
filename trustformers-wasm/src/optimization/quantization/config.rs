//! Quantization configuration and supporting types

use serde::{Deserialize, Serialize};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Quantization strategies
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// No quantization
    None,
    /// Dynamic quantization - quantize weights only
    Dynamic,
    /// Static quantization - quantize weights and activations
    Static,
    /// Post-training quantization
    PostTraining,
    /// Quantization-aware training (requires pre-quantized model)
    QAT,
    /// AWQ (Activation-aware Weight Quantization) - preserves important weights
    AWQ,
    /// GPTQ (Gradient-based Post-Training Quantization) - uses second-order information
    GPTQ,
    /// SmoothQuant - balances weights and activations difficulty
    SmoothQuant,
    /// LLM.int8() - mixed-precision quantization for large models
    LLMInt8,
    /// QLoRA - Quantized Low-Rank Adaptation
    QLoRA,
    /// GGML-style quantization for efficient inference
    GGML,
    /// Adaptive bitwidth quantization with dynamic allocation
    AdaptiveBitwidth,
    /// Outlier-aware quantization for handling activation spikes
    OutlierAware,
    /// HQQ (Half-Quadratic Quantization) - superior quality quantization using half-quadratic optimization
    HQQ,
    /// SpQR (Sparse-Quantized Representation) - ultra-sparse models with mixed precision
    SpQR,
    /// AQLM (Additive Quantization for Language Models) - additive quantization for transformers
    AQLM,
}

/// Quantization precision levels
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    /// 16-bit floating point
    FP16,
    /// 8-bit floating point (E4M3 or E5M2 format)
    FP8,
    /// 8-bit integer
    INT8,
    /// 4-bit integer
    INT4,
    /// 2-bit integer (experimental)
    INT2,
    /// 1-bit binary quantization
    INT1,
    /// Mixed precision (FP16 for outliers, INT8 for normal weights)
    Mixed,
    /// Adaptive precision based on layer importance
    Adaptive,
}

/// Quantization configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    strategy: QuantizationStrategy,
    precision: QuantizationPrecision,
    target_size_mb: f32,
    performance_threshold: f32,
    accuracy_threshold: f32,
    auto_select: bool,
}

#[wasm_bindgen]
impl QuantizationConfig {
    /// Create a new quantization configuration
    #[wasm_bindgen(constructor)]
    pub fn new(strategy: QuantizationStrategy, precision: QuantizationPrecision) -> Self {
        Self {
            strategy,
            precision,
            target_size_mb: 50.0,       // Default target size
            performance_threshold: 2.0, // 2x speedup minimum
            accuracy_threshold: 0.95,   // 95% accuracy retention minimum
            auto_select: false,
        }
    }

    /// Create an automatic configuration that selects best settings
    pub fn auto() -> Self {
        Self {
            strategy: QuantizationStrategy::Dynamic,
            precision: QuantizationPrecision::INT8,
            target_size_mb: 10.0,
            performance_threshold: 1.5,
            accuracy_threshold: 0.90,
            auto_select: true,
        }
    }

    /// Create a configuration optimized for mobile devices
    pub fn mobile() -> Self {
        Self {
            strategy: QuantizationStrategy::PostTraining,
            precision: QuantizationPrecision::INT8,
            target_size_mb: 5.0,
            performance_threshold: 3.0,
            accuracy_threshold: 0.85,
            auto_select: false,
        }
    }

    /// Create a configuration for desktop/high-performance devices
    pub fn desktop() -> Self {
        Self {
            strategy: QuantizationStrategy::Dynamic,
            precision: QuantizationPrecision::FP16,
            target_size_mb: 100.0,
            performance_threshold: 1.2,
            accuracy_threshold: 0.98,
            auto_select: false,
        }
    }

    /// Create a configuration for ultra-low latency inference
    pub fn ultra_fast() -> Self {
        Self {
            strategy: QuantizationStrategy::GGML,
            precision: QuantizationPrecision::FP8,
            target_size_mb: 15.0,
            performance_threshold: 4.0,
            accuracy_threshold: 0.88,
            auto_select: false,
        }
    }

    /// Create a configuration for fine-tuning with QLoRA
    pub fn qlora() -> Self {
        Self {
            strategy: QuantizationStrategy::QLoRA,
            precision: QuantizationPrecision::Mixed,
            target_size_mb: 8.0,
            performance_threshold: 2.5,
            accuracy_threshold: 0.92,
            auto_select: false,
        }
    }

    /// Create a configuration with adaptive bitwidth for optimal efficiency
    pub fn adaptive() -> Self {
        Self {
            strategy: QuantizationStrategy::AdaptiveBitwidth,
            precision: QuantizationPrecision::Adaptive,
            target_size_mb: 12.0,
            performance_threshold: 3.0,
            accuracy_threshold: 0.93,
            auto_select: true,
        }
    }

    /// Create a configuration for models with activation outliers
    pub fn outlier_aware() -> Self {
        Self {
            strategy: QuantizationStrategy::OutlierAware,
            precision: QuantizationPrecision::Mixed,
            target_size_mb: 20.0,
            performance_threshold: 2.0,
            accuracy_threshold: 0.96,
            auto_select: false,
        }
    }

    /// Set target model size in MB
    pub fn set_target_size_mb(mut self, size_mb: f32) -> Self {
        self.target_size_mb = size_mb;
        self
    }

    /// Set performance threshold (minimum speedup factor)
    pub fn set_performance_threshold(mut self, threshold: f32) -> Self {
        self.performance_threshold = threshold;
        self
    }

    /// Set accuracy threshold (minimum accuracy retention)
    pub fn set_accuracy_threshold(mut self, threshold: f32) -> Self {
        self.accuracy_threshold = threshold;
        self
    }

    /// Enable automatic strategy selection
    pub fn enable_auto_select(mut self) -> Self {
        self.auto_select = true;
        self
    }

    // Getters for private fields
    #[wasm_bindgen(getter)]
    pub fn strategy(&self) -> QuantizationStrategy {
        self.strategy
    }

    #[wasm_bindgen(getter)]
    pub fn precision(&self) -> QuantizationPrecision {
        self.precision
    }

    #[wasm_bindgen(getter)]
    pub fn target_size_mb(&self) -> f32 {
        self.target_size_mb
    }

    #[wasm_bindgen(getter)]
    pub fn performance_threshold(&self) -> f32 {
        self.performance_threshold
    }

    #[wasm_bindgen(getter)]
    pub fn accuracy_threshold(&self) -> f32 {
        self.accuracy_threshold
    }

    #[wasm_bindgen(getter)]
    pub fn auto_select(&self) -> bool {
        self.auto_select
    }
}

/// Quantization statistics
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    original_size_bytes: usize,
    quantized_size_bytes: usize,
    compression_ratio: f32,
    size_reduction_percent: f32,
    estimated_speedup: f32,
    strategy_used: QuantizationStrategy,
    precision_used: QuantizationPrecision,
}

#[wasm_bindgen]
impl QuantizationStats {
    #[wasm_bindgen(getter)]
    pub fn original_size_bytes(&self) -> usize {
        self.original_size_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn quantized_size_bytes(&self) -> usize {
        self.quantized_size_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn compression_ratio(&self) -> f32 {
        self.compression_ratio
    }

    #[wasm_bindgen(getter)]
    pub fn size_reduction_percent(&self) -> f32 {
        self.size_reduction_percent
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_speedup(&self) -> f32 {
        self.estimated_speedup
    }

    #[wasm_bindgen(getter)]
    pub fn strategy_used(&self) -> QuantizationStrategy {
        self.strategy_used
    }

    #[wasm_bindgen(getter)]
    pub fn precision_used(&self) -> QuantizationPrecision {
        self.precision_used
    }
}

impl QuantizationStats {
    /// Create new quantization statistics
    pub fn new(
        original_size_bytes: usize,
        quantized_size_bytes: usize,
        compression_ratio: f32,
        size_reduction_percent: f32,
        estimated_speedup: f32,
        strategy_used: QuantizationStrategy,
        precision_used: QuantizationPrecision,
    ) -> Self {
        Self {
            original_size_bytes,
            quantized_size_bytes,
            compression_ratio,
            size_reduction_percent,
            estimated_speedup,
            strategy_used,
            precision_used,
        }
    }
}

/// Runtime performance monitoring for adaptive quantization
#[derive(Debug, Clone)]
pub struct RuntimeMonitor {
    pub inference_times: Vec<f64>,
    pub memory_usage: Vec<usize>,
    pub accuracy_scores: Vec<f32>,
    pub thermal_state: ThermalState,
    pub adaptation_history: Vec<AdaptationEvent>,
}

/// Current device thermal state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal,  // Normal operating temperature
    Fair,     // Slightly elevated
    Serious,  // High temperature, throttling recommended
    Critical, // Very high temperature, aggressive throttling needed
}

/// Adaptive quantization state that changes at runtime
#[derive(Debug, Clone)]
pub struct AdaptiveQuantizationState {
    pub current_strategy: QuantizationStrategy,
    pub current_precision: QuantizationPrecision,
    pub adaptation_rate: f32,
    pub performance_target: f32,
    pub accuracy_target: f32,
    pub last_adaptation: f64,
    pub confidence_score: f32,
}

/// Record of quantization adaptations
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    pub timestamp: f64,
    pub trigger: AdaptationTrigger,
    pub old_strategy: QuantizationStrategy,
    pub new_strategy: QuantizationStrategy,
    pub old_precision: QuantizationPrecision,
    pub new_precision: QuantizationPrecision,
    pub improvement_ratio: f32,
}

/// What triggered the adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationTrigger {
    PerformanceDrop,     // Inference too slow
    MemoryPressure,      // Running out of memory
    AccuracyDrop,        // Model accuracy below threshold
    ThermalThrottling,   // Device overheating
    BatteryOptimization, // Low battery, need efficiency
    WorkloadChange,      // Different type of inference requests
}

/// Device capabilities for quantization optimization
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub supports_int8: bool,
    pub supports_int4: bool,
    pub supports_fp16: bool,
    pub memory_bandwidth_gb_s: f32,
    pub compute_capability: ComputeCapability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeCapability {
    Low,    // Basic CPU
    Medium, // High-end CPU or integrated GPU
    High,   // Dedicated GPU
}
