//! Web-optimized quantizer with runtime adaptation

use crate::optimization::quantization::algorithms::*;
use crate::optimization::quantization::config::*;
use serde::{Deserialize, Serialize};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Result of quantization operation
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    data: Vec<u8>,
    stats: QuantizationStats,
}

#[wasm_bindgen]
impl QuantizationResult {
    /// Get the quantized data
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Get a summary of the quantization
    pub fn summary(&self) -> String {
        format!(
            "Quantization completed: {:.1}x compression, {:.1}% size reduction, {:.1}x estimated speedup",
            self.stats.compression_ratio(),
            self.stats.size_reduction_percent(),
            self.stats.estimated_speedup()
        )
    }

    /// Get detailed statistics
    pub fn stats(&self) -> QuantizationStats {
        self.stats.clone()
    }
}

/// Web-optimized quantizer with runtime adaptation
#[wasm_bindgen]
pub struct WebQuantizer {
    config: QuantizationConfig,
    #[allow(dead_code)]
    device_capabilities: DeviceCapabilities,
    #[allow(dead_code)]
    runtime_monitor: RuntimeMonitor,
    adaptive_state: AdaptiveQuantizationState,
}

#[wasm_bindgen]
impl WebQuantizer {
    /// Create a new web quantizer
    #[wasm_bindgen(constructor)]
    pub fn new(config: QuantizationConfig) -> Self {
        let device_capabilities = DeviceCapabilities {
            supports_int8: true,
            supports_int4: true,
            supports_fp16: true,
            memory_bandwidth_gb_s: 100.0,
            compute_capability: ComputeCapability::Medium,
        };

        let runtime_monitor = RuntimeMonitor {
            inference_times: Vec::new(),
            memory_usage: Vec::new(),
            accuracy_scores: Vec::new(),
            thermal_state: ThermalState::Nominal,
            adaptation_history: Vec::new(),
        };

        let adaptive_state = AdaptiveQuantizationState {
            current_strategy: config.strategy(),
            current_precision: config.precision(),
            adaptation_rate: 0.1,
            performance_target: config.performance_threshold(),
            accuracy_target: config.accuracy_threshold(),
            last_adaptation: 0.0,
            confidence_score: 0.8,
        };

        Self {
            config,
            device_capabilities,
            runtime_monitor,
            adaptive_state,
        }
    }

    /// Quantize tensor data using the configured strategy
    pub fn quantize(&self, data: &[f32]) -> Result<Vec<f32>, JsValue> {
        match self.adaptive_state.current_strategy {
            QuantizationStrategy::None => Ok(data.to_vec()),
            QuantizationStrategy::Dynamic => {
                apply_dynamic_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::Static => {
                apply_static_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::PostTraining => {
                apply_post_training_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::AWQ => {
                apply_awq_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::GPTQ => {
                apply_gptq_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::SmoothQuant => {
                apply_smoothquant_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::LLMInt8 => {
                apply_llm_int8_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::QLoRA => {
                apply_qlora_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::GGML => {
                apply_ggml_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::AdaptiveBitwidth => {
                apply_adaptive_bitwidth_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::OutlierAware => {
                apply_outlier_aware_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::HQQ => {
                apply_hqq_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::SpQR => {
                apply_spqr_quantization(data, self.adaptive_state.current_precision)
            },
            QuantizationStrategy::AQLM => {
                apply_aqlm_quantization(data, self.adaptive_state.current_precision)
            },
            _ => Err(JsValue::from_str("Unsupported quantization strategy")),
        }
    }

    /// Get quantization statistics
    pub fn get_stats(&self, original_data: &[f32], quantized_data: &[f32]) -> QuantizationStats {
        let original_size = original_data.len() * 4; // 4 bytes per f32
        let quantized_size = quantized_data.len() * 4; // Simplified for placeholder
        let compression_ratio = original_size as f32 / quantized_size as f32;
        let size_reduction = (1.0 - quantized_size as f32 / original_size as f32) * 100.0;

        QuantizationStats::new(
            original_size,
            quantized_size,
            compression_ratio,
            size_reduction,
            compression_ratio * 0.8, // Simplified estimation
            self.adaptive_state.current_strategy,
            self.adaptive_state.current_precision,
        )
    }

    /// Check if a model should be quantized based on size
    pub fn should_quantize(&self, model_size_bytes: usize) -> bool {
        let model_size_mb = model_size_bytes as f32 / (1024.0 * 1024.0);
        model_size_mb > self.config.target_size_mb()
    }

    /// Quantize model data
    pub fn quantize_model(&self, model_data: &[u8]) -> Result<QuantizationResult, JsValue> {
        // Convert bytes to f32 for processing
        let float_data: Vec<f32> = model_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Apply quantization
        let quantized_floats = self.quantize(&float_data)?;

        // Get stats before converting to bytes
        let stats = self.get_stats(&float_data, &quantized_floats);

        // Convert back to bytes
        let quantized_bytes: Vec<u8> =
            quantized_floats.into_iter().flat_map(|f| f.to_le_bytes()).collect();

        Ok(QuantizationResult {
            data: quantized_bytes,
            stats,
        })
    }

    /// Get recommended quantization settings for a given model size
    pub fn get_recommended_settings(&self, model_size_bytes: usize) -> QuantizationConfig {
        let model_size_mb = model_size_bytes as f32 / (1024.0 * 1024.0);

        if model_size_mb < 10.0 {
            QuantizationConfig::new(QuantizationStrategy::None, QuantizationPrecision::FP16)
        } else if model_size_mb < 50.0 {
            QuantizationConfig::new(QuantizationStrategy::Dynamic, QuantizationPrecision::FP16)
        } else if model_size_mb < 200.0 {
            QuantizationConfig::new(
                QuantizationStrategy::PostTraining,
                QuantizationPrecision::INT8,
            )
        } else {
            QuantizationConfig::new(QuantizationStrategy::AWQ, QuantizationPrecision::INT4)
        }
    }
}

/// Quantized model data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedModelData {
    pub quantized_weights: Vec<Vec<f32>>,
    pub scale_factors: Vec<f32>,
    pub zero_points: Vec<f32>,
    pub metadata: QuantizationMetadata,
}

/// Quantization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    pub strategy: QuantizationStrategy,
    pub precision: QuantizationPrecision,
    pub compression_ratio: f32,
    pub accuracy_retention: f32,
}

impl QuantizedModelData {
    pub fn new(strategy: QuantizationStrategy, precision: QuantizationPrecision) -> Self {
        Self {
            quantized_weights: Vec::new(),
            scale_factors: Vec::new(),
            zero_points: Vec::new(),
            metadata: QuantizationMetadata {
                strategy,
                precision,
                compression_ratio: 1.0,
                accuracy_retention: 1.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_creation() {
        let config = QuantizationConfig::auto();
        let quantizer = WebQuantizer::new(config);
        assert_eq!(
            quantizer.adaptive_state.current_strategy,
            QuantizationStrategy::Dynamic
        );
    }

    #[test]
    fn test_basic_quantization() {
        let config =
            QuantizationConfig::new(QuantizationStrategy::Dynamic, QuantizationPrecision::INT8);
        let quantizer = WebQuantizer::new(config);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = quantizer.quantize(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantization_stats() {
        let config = QuantizationConfig::auto();
        let quantizer = WebQuantizer::new(config);
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let quantized = vec![0.5, 1.0, 1.5, 2.0];
        let stats = quantizer.get_stats(&original, &quantized);
        assert!(stats.compression_ratio() >= 1.0);
    }

    #[test]
    fn test_quantized_model_data() {
        let data = QuantizedModelData::new(QuantizationStrategy::AWQ, QuantizationPrecision::INT8);
        assert_eq!(data.metadata.strategy, QuantizationStrategy::AWQ);
        assert_eq!(data.metadata.precision, QuantizationPrecision::INT8);
    }
}
