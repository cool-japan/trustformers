//! Quantization utility functions and helpers
//!
//! This module contains utility functions for quantization configuration recommendation,
//! validation, impact estimation, and other helper functionality.

use super::config::QuantizationConfig;
use super::types::*;
use crate::error::TrustformersResult;

/// Utility functions for quantization
pub struct QuantizationUtils;

/// Quantization impact estimation results
#[derive(Debug, Clone)]
pub struct QuantizationImpactEstimate {
    /// Expected compression ratio
    pub compression_ratio: f64,
    /// Estimated accuracy drop (percentage)
    pub estimated_accuracy_drop: f64,
    /// Estimated inference speedup
    pub estimated_speedup: f64,
    /// Memory savings in MB
    pub memory_savings_mb: f64,
    /// Estimated quantization time in minutes
    pub quantization_time_estimate_minutes: f64,
}

impl QuantizationUtils {
    /// Recommend quantization configuration for a model
    pub fn recommend_quantization_config(
        model_size_mb: f64,
        target_platform: &str,
        accuracy_target: f64,
    ) -> QuantizationConfig {
        let mut config = QuantizationConfig::default();

        // Determine optimal quantization method based on constraints
        if model_size_mb > 1000.0 {
            // Large models (>1GB)
            if accuracy_target > 99.0 {
                config.quantization_type = QuantizationType::AWQ;
                config.weight_precision = QuantizationPrecision::INT4;
            } else if accuracy_target > 95.0 {
                config.quantization_type = QuantizationType::GPTQ;
                config.weight_precision = QuantizationPrecision::INT4;
            } else {
                config.quantization_type = QuantizationType::INT4;
            }
        } else if model_size_mb > 100.0 {
            // Medium models (100MB-1GB)
            if accuracy_target >= 98.0 {
                config.quantization_type = QuantizationType::INT8;
            } else {
                config.quantization_type = QuantizationType::Dynamic;
                config.weight_precision = QuantizationPrecision::INT8;
            }
        } else {
            // Small models (<100MB)
            config.quantization_type = QuantizationType::MixedPrecision;
            config.weight_precision = QuantizationPrecision::FP16;
            config.activation_precision = QuantizationPrecision::FP16;
        }

        // Platform-specific optimizations
        match target_platform.to_lowercase().as_str() {
            "cpu" => {
                config.performance_settings.optimized_kernels = true;
                config.granularity = QuantizationGranularity::PerChannel;
            },
            "gpu" => {
                config.performance_settings.kernel_fusion = true;
                config.granularity = QuantizationGranularity::PerTensor;
            },
            "mobile" => {
                config.quantization_type = QuantizationType::INT8;
                config.performance_settings.memory_optimization = 3;
            },
            "edge" => {
                config.quantization_type = QuantizationType::INT4;
                config.performance_settings.cache_quantized_weights = false;
            },
            _ => {
                // Default configuration
            },
        }

        config
    }

    /// Validate quantization configuration
    pub fn validate_config(config: &QuantizationConfig) -> TrustformersResult<Vec<String>> {
        let mut warnings = Vec::new();

        // Check for incompatible settings
        if config.quantization_type == QuantizationType::Dynamic && config.quantize_activations {
            warnings.push("Dynamic quantization automatically handles activations, activation quantization flag will be ignored".to_string());
        }

        if config.weight_precision == QuantizationPrecision::INT4
            && config.granularity == QuantizationGranularity::PerTensor
        {
            warnings.push("INT4 quantization typically works better with per-channel or per-group granularity".to_string());
        }

        if config.calibration_samples < 128 {
            warnings.push("Calibration sample count is low, consider using at least 128 samples for better accuracy".to_string());
        }

        if config.advanced_settings.outlier_percentile > 99.99 {
            warnings.push(
                "Very high outlier percentile may not provide effective outlier detection"
                    .to_string(),
            );
        }

        // Check for missing calibration dataset for methods that require it
        if matches!(
            config.quantization_type,
            QuantizationType::GPTQ | QuantizationType::AWQ | QuantizationType::SmoothQuant
        ) && config.calibration_dataset.is_none()
        {
            warnings.push("Selected quantization method requires calibration dataset".to_string());
        }

        // Check memory optimization level
        if config.performance_settings.memory_optimization > 3 {
            warnings.push(
                "Memory optimization level > 3 may significantly impact performance".to_string(),
            );
        }

        // Check thread count
        if config.performance_settings.num_threads > num_cpus::get() as u32 * 2 {
            warnings.push(
                "Thread count exceeds 2x CPU cores, may cause performance degradation".to_string(),
            );
        }

        Ok(warnings)
    }

    /// Estimate quantization impact
    pub fn estimate_impact(
        config: &QuantizationConfig,
        model_size_mb: f64,
    ) -> QuantizationImpactEstimate {
        let compression_ratio = match config.quantization_type {
            QuantizationType::INT8 => 4.0,
            QuantizationType::INT4 => 8.0,
            QuantizationType::Dynamic => 3.5,
            QuantizationType::MixedPrecision => 2.0,
            QuantizationType::QAT => match config.weight_precision {
                QuantizationPrecision::INT8 => 4.0,
                QuantizationPrecision::INT4 => 8.0,
                _ => 2.0,
            },
            QuantizationType::GPTQ => match config.weight_precision {
                QuantizationPrecision::INT4 => 7.5,
                QuantizationPrecision::INT8 => 3.8,
                _ => 4.0,
            },
            QuantizationType::AWQ => match config.weight_precision {
                QuantizationPrecision::INT4 => 7.8,
                QuantizationPrecision::INT8 => 3.9,
                _ => 4.0,
            },
            QuantizationType::SmoothQuant => 3.8,
            QuantizationType::GGML => match config.weight_precision {
                QuantizationPrecision::INT4 => 8.0,
                QuantizationPrecision::INT8 => 4.0,
                _ => 2.0,
            },
            QuantizationType::Custom => 2.0, // Conservative estimate
            QuantizationType::None => 1.0,
        };

        let estimated_accuracy_drop = match config.quantization_type {
            QuantizationType::INT8 => 0.5,
            QuantizationType::INT4 => 2.0,
            QuantizationType::Dynamic => 0.3,
            QuantizationType::MixedPrecision => 0.1,
            QuantizationType::QAT => match config.weight_precision {
                QuantizationPrecision::INT8 => 0.2,
                QuantizationPrecision::INT4 => 1.0,
                _ => 0.1,
            },
            QuantizationType::GPTQ => match config.weight_precision {
                QuantizationPrecision::INT4 => 1.5,
                QuantizationPrecision::INT8 => 0.4,
                _ => 0.5,
            },
            QuantizationType::AWQ => match config.weight_precision {
                QuantizationPrecision::INT4 => 1.2,
                QuantizationPrecision::INT8 => 0.3,
                _ => 0.4,
            },
            QuantizationType::SmoothQuant => 0.4,
            QuantizationType::GGML => match config.weight_precision {
                QuantizationPrecision::INT4 => 2.5,
                QuantizationPrecision::INT8 => 0.6,
                _ => 1.0,
            },
            QuantizationType::Custom => 1.0, // Conservative estimate
            QuantizationType::None => 0.0,
        };

        let estimated_speedup = match config.quantization_type {
            QuantizationType::INT8 => 2.5,
            QuantizationType::INT4 => 4.0,
            QuantizationType::Dynamic => 2.0,
            QuantizationType::MixedPrecision => 1.5,
            QuantizationType::QAT => match config.weight_precision {
                QuantizationPrecision::INT8 => 2.5,
                QuantizationPrecision::INT4 => 4.0,
                _ => 1.5,
            },
            QuantizationType::GPTQ => match config.weight_precision {
                QuantizationPrecision::INT4 => 3.8,
                QuantizationPrecision::INT8 => 2.4,
                _ => 2.0,
            },
            QuantizationType::AWQ => match config.weight_precision {
                QuantizationPrecision::INT4 => 3.9,
                QuantizationPrecision::INT8 => 2.5,
                _ => 2.0,
            },
            QuantizationType::SmoothQuant => 2.3,
            QuantizationType::GGML => match config.weight_precision {
                QuantizationPrecision::INT4 => 4.2,
                QuantizationPrecision::INT8 => 2.6,
                _ => 1.8,
            },
            QuantizationType::Custom => 1.5, // Conservative estimate
            QuantizationType::None => 1.0,
        };

        let memory_savings_mb = model_size_mb * (compression_ratio - 1.0) / compression_ratio;

        // Estimate quantization time based on model size and method complexity
        let base_time_minutes = model_size_mb / 100.0; // Base: 1 minute per 100MB
        let complexity_multiplier = match config.quantization_type {
            QuantizationType::INT8 => 1.0,
            QuantizationType::INT4 => 1.2,
            QuantizationType::Dynamic => 0.8,
            QuantizationType::MixedPrecision => 1.5,
            QuantizationType::QAT => 10.0, // QAT requires training
            QuantizationType::GPTQ => 3.0,
            QuantizationType::AWQ => 2.5,
            QuantizationType::SmoothQuant => 2.0,
            QuantizationType::GGML => 1.1,
            QuantizationType::Custom => 2.0,
            QuantizationType::None => 0.1,
        };

        let quantization_time_estimate_minutes = base_time_minutes * complexity_multiplier;

        QuantizationImpactEstimate {
            compression_ratio,
            estimated_accuracy_drop,
            estimated_speedup,
            memory_savings_mb,
            quantization_time_estimate_minutes,
        }
    }

    /// Get optimal calibration sample count for a given model size
    pub fn get_optimal_calibration_samples(model_size_mb: f64) -> u32 {
        if model_size_mb > 1000.0 {
            1000 // Large models need more samples
        } else if model_size_mb > 100.0 {
            500 // Medium models
        } else {
            128 // Small models
        }
    }

    /// Check if quantization method is supported on target platform
    pub fn is_method_supported(quantization_type: QuantizationType, target_platform: &str) -> bool {
        match target_platform.to_lowercase().as_str() {
            "cpu" => {
                // All methods supported on CPU
                true
            },
            "gpu" => {
                // Most methods supported on GPU
                !matches!(quantization_type, QuantizationType::GGML)
            },
            "mobile" => {
                // Limited methods for mobile
                matches!(
                    quantization_type,
                    QuantizationType::INT8 | QuantizationType::INT4 | QuantizationType::Dynamic
                )
            },
            "edge" => {
                // Very limited methods for edge devices
                matches!(
                    quantization_type,
                    QuantizationType::INT8 | QuantizationType::INT4
                )
            },
            _ => false,
        }
    }

    /// Get recommended granularity for quantization method
    pub fn get_recommended_granularity(
        quantization_type: QuantizationType,
        weight_precision: QuantizationPrecision,
    ) -> QuantizationGranularity {
        match quantization_type {
            QuantizationType::INT4 | QuantizationType::GPTQ | QuantizationType::AWQ => {
                QuantizationGranularity::PerChannel
            },
            QuantizationType::INT8 => match weight_precision {
                QuantizationPrecision::INT4 => QuantizationGranularity::PerChannel,
                _ => QuantizationGranularity::PerTensor,
            },
            QuantizationType::Dynamic | QuantizationType::SmoothQuant => {
                QuantizationGranularity::PerChannel
            },
            _ => QuantizationGranularity::PerTensor,
        }
    }

    /// Adjust configuration for target memory budget
    pub fn adjust_for_memory_budget(
        mut config: QuantizationConfig,
        model_size_mb: f64,
        target_memory_mb: f64,
    ) -> QuantizationConfig {
        let required_compression = model_size_mb / target_memory_mb;

        if required_compression <= 2.0 {
            // Light compression
            config.quantization_type = QuantizationType::MixedPrecision;
            config.weight_precision = QuantizationPrecision::FP16;
        } else if required_compression <= 4.0 {
            // Moderate compression
            config.quantization_type = QuantizationType::INT8;
            config.weight_precision = QuantizationPrecision::INT8;
        } else if required_compression <= 8.0 {
            // High compression
            config.quantization_type = QuantizationType::AWQ;
            config.weight_precision = QuantizationPrecision::INT4;
        } else {
            // Extreme compression
            config.quantization_type = QuantizationType::INT4;
            config.weight_precision = QuantizationPrecision::INT4;
            config.advanced_settings.mixed_bit_config = Some(Default::default());
        }

        config
    }
}
