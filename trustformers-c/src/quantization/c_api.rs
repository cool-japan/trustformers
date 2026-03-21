//! C API for quantization functionality
//!
//! This module provides C-compatible functions for using the quantization system
//! from external languages and applications.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;
use parking_lot::RwLock;

use crate::error::TrustformersError;
use crate::utils::{c_str_to_string, str_to_c_str, string_to_c_str};

use super::config::*;
use super::engine::*;
use super::types::*;

/// Handle type for quantization engines
pub type QuantizationHandle = usize;

/// Global storage for quantization engines
static QUANTIZATION_ENGINES: Lazy<RwLock<HashMap<QuantizationHandle, QuantizationEngine>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Next available handle
static NEXT_QUANTIZATION_HANDLE: AtomicUsize = AtomicUsize::new(1);

/// C-compatible quantization configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersQuantizationConfig {
    /// Quantization type
    pub quantization_type: c_int,
    /// Weight precision bits
    pub weight_bits: c_int,
    /// Activation precision bits
    pub activation_bits: c_int,
    /// Whether to quantize weights
    pub quantize_weights: c_int,
    /// Whether to quantize activations
    pub quantize_activations: c_int,
    /// Calibration method
    pub calibration_method: c_int,
    /// Number of calibration samples
    pub calibration_samples: c_int,
    /// Calibration dataset path
    pub calibration_dataset_path: *const c_char,
    /// Use symmetric quantization
    pub symmetric_quantization: c_int,
    /// Enable advanced optimizations
    pub enable_advanced_optimizations: c_int,
    /// Number of quantization threads
    pub num_threads: c_int,
}

/// C-compatible quantization statistics
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersQuantizationStats {
    /// Original model size in bytes
    pub original_size_bytes: u64,
    /// Quantized model size in bytes
    pub quantized_size_bytes: u64,
    /// Compression ratio
    pub compression_ratio: c_double,
    /// Memory savings in bytes
    pub memory_savings: u64,
    /// Accuracy impact percentage
    pub accuracy_impact: c_double,
    /// Inference speedup factor
    pub speedup_factor: c_double,
    /// Quantization time in seconds
    pub quantization_time_seconds: c_double,
}

/// Create a new quantization engine
#[no_mangle]
pub extern "C" fn trustformers_quantization_create_engine(
    config: *const TrustformersQuantizationConfig,
    handle: *mut QuantizationHandle,
) -> TrustformersError {
    if config.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let c_config = unsafe { &*config };

    // Convert C config to Rust config
    let quantization_type = match c_config.quantization_type {
        0 => QuantizationType::None,
        1 => QuantizationType::INT8,
        2 => QuantizationType::INT4,
        3 => QuantizationType::Dynamic,
        4 => QuantizationType::MixedPrecision,
        5 => QuantizationType::QAT,
        6 => QuantizationType::GPTQ,
        7 => QuantizationType::AWQ,
        8 => QuantizationType::SmoothQuant,
        9 => QuantizationType::GGML,
        99 => QuantizationType::Custom,
        _ => return TrustformersError::InvalidParameter,
    };

    let weight_precision = match c_config.weight_bits {
        32 => QuantizationPrecision::FP32,
        16 => QuantizationPrecision::FP16,
        15 => QuantizationPrecision::BF16,
        8 => QuantizationPrecision::INT8,
        4 => QuantizationPrecision::INT4,
        2 => QuantizationPrecision::INT2,
        1 => QuantizationPrecision::INT1,
        _ => return TrustformersError::InvalidParameter,
    };

    let activation_precision = match c_config.activation_bits {
        32 => QuantizationPrecision::FP32,
        16 => QuantizationPrecision::FP16,
        15 => QuantizationPrecision::BF16,
        8 => QuantizationPrecision::INT8,
        4 => QuantizationPrecision::INT4,
        2 => QuantizationPrecision::INT2,
        1 => QuantizationPrecision::INT1,
        _ => return TrustformersError::InvalidParameter,
    };

    let calibration_method = match c_config.calibration_method {
        0 => CalibrationMethod::MinMax,
        1 => CalibrationMethod::Entropy,
        2 => CalibrationMethod::Percentile,
        3 => CalibrationMethod::MeanStd,
        4 => CalibrationMethod::Histogram,
        _ => return TrustformersError::InvalidParameter,
    };

    let calibration_dataset = if c_config.calibration_dataset_path.is_null() {
        None
    } else {
        match c_str_to_string(c_config.calibration_dataset_path) {
            Ok(path) => Some(path),
            Err(_) => return TrustformersError::InvalidParameter,
        }
    };

    let rust_config = QuantizationConfig {
        quantization_type,
        weight_precision,
        activation_precision,
        quantize_weights: c_config.quantize_weights != 0,
        quantize_activations: c_config.quantize_activations != 0,
        calibration_method,
        calibration_dataset,
        calibration_samples: c_config.calibration_samples as u32,
        range_settings: RangeSettings {
            symmetric: c_config.symmetric_quantization != 0,
            ..Default::default()
        },
        performance_settings: QuantizationPerformanceSettings {
            num_threads: if c_config.num_threads > 0 {
                c_config.num_threads as u32
            } else {
                num_cpus::get() as u32
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let engine = QuantizationEngine::new(rust_config);
    let engine_handle = NEXT_QUANTIZATION_HANDLE.fetch_add(1, Ordering::Relaxed);

    QUANTIZATION_ENGINES.write().insert(engine_handle, engine);

    unsafe {
        *handle = engine_handle;
    }

    TrustformersError::Success
}

/// Set calibration data for quantization
#[no_mangle]
pub extern "C" fn trustformers_quantization_set_calibration_data(
    handle: QuantizationHandle,
    data: *const c_float,
    data_len: usize,
    shape: *const i64,
    shape_len: usize,
    num_samples: usize,
) -> TrustformersError {
    if data.is_null() || shape.is_null() {
        return TrustformersError::NullPointer;
    }

    let mut engines = QUANTIZATION_ENGINES.write();
    let engine = match engines.get_mut(&handle) {
        Some(engine) => engine,
        None => return TrustformersError::InvalidHandle,
    };

    // Convert C data to calibration samples
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len) };
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, shape_len) };

    let sample_size = data_len / num_samples;
    let mut samples = Vec::new();

    for i in 0..num_samples {
        let start_idx = i * sample_size;
        let end_idx = (i + 1) * sample_size;
        let sample_data = data_slice[start_idx..end_idx].to_vec();

        let sample = CalibrationSample {
            input_data: sample_data,
            input_shape: shape_slice.to_vec(),
            weight: 1.0,
        };
        samples.push(sample);
    }

    match engine.set_calibration_data(samples) {
        Ok(_) => TrustformersError::Success,
        Err(err) => err,
    }
}

/// Run calibration on the quantization engine
#[no_mangle]
pub extern "C" fn trustformers_quantization_calibrate(
    handle: QuantizationHandle,
) -> TrustformersError {
    let mut engines = QUANTIZATION_ENGINES.write();
    let engine = match engines.get_mut(&handle) {
        Some(engine) => engine,
        None => return TrustformersError::InvalidHandle,
    };

    match engine.calibrate() {
        Ok(_) => TrustformersError::Success,
        Err(err) => err,
    }
}

/// Quantize a model using the configured quantization engine
#[no_mangle]
pub extern "C" fn trustformers_quantization_quantize_model(
    handle: QuantizationHandle,
    model_path: *const c_char,
    output_path: *mut *mut c_char,
) -> TrustformersError {
    if model_path.is_null() || output_path.is_null() {
        return TrustformersError::NullPointer;
    }

    let model_path_str = match c_str_to_string(model_path) {
        Ok(path) => path,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut engines = QUANTIZATION_ENGINES.write();
    let engine = match engines.get_mut(&handle) {
        Some(engine) => engine,
        None => return TrustformersError::InvalidHandle,
    };

    match engine.quantize_model(&model_path_str) {
        Ok(quantized_path) => {
            let c_str = string_to_c_str(quantized_path);
            if c_str.is_null() {
                TrustformersError::OutOfMemory
            } else {
                unsafe {
                    *output_path = c_str;
                }
                TrustformersError::Success
            }
        },
        Err(err) => err,
    }
}

/// Get quantization statistics
#[no_mangle]
pub extern "C" fn trustformers_quantization_get_stats(
    handle: QuantizationHandle,
    stats: *mut TrustformersQuantizationStats,
) -> TrustformersError {
    if stats.is_null() {
        return TrustformersError::NullPointer;
    }

    let engines = QUANTIZATION_ENGINES.read();
    let engine = match engines.get(&handle) {
        Some(engine) => engine,
        None => return TrustformersError::InvalidHandle,
    };

    let engine_stats = match engine.get_stats() {
        Some(stats) => stats,
        None => return TrustformersError::RuntimeError,
    };

    unsafe {
        (*stats).original_size_bytes = engine_stats.original_size_bytes;
        (*stats).quantized_size_bytes = engine_stats.quantized_size_bytes;
        (*stats).compression_ratio = engine_stats.compression_ratio;
        (*stats).memory_savings = engine_stats.memory_savings;
        (*stats).accuracy_impact = engine_stats.accuracy_impact;
        (*stats).speedup_factor = engine_stats.speedup_factor;
        (*stats).quantization_time_seconds = engine_stats.quantization_time_seconds;
    }

    TrustformersError::Success
}

/// Export quantized model to specified format
#[no_mangle]
pub extern "C" fn trustformers_quantization_export_model(
    handle: QuantizationHandle,
    quantized_model_path: *const c_char,
    export_format: *const c_char,
    output_path: *const c_char,
) -> TrustformersError {
    if quantized_model_path.is_null() || export_format.is_null() || output_path.is_null() {
        return TrustformersError::NullPointer;
    }

    let quantized_path_str = match c_str_to_string(quantized_model_path) {
        Ok(path) => path,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let format_str = match c_str_to_string(export_format) {
        Ok(format) => format,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let output_path_str = match c_str_to_string(output_path) {
        Ok(path) => path,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let engines = QUANTIZATION_ENGINES.read();
    let engine = match engines.get(&handle) {
        Some(engine) => engine,
        None => return TrustformersError::InvalidHandle,
    };

    match engine.export_quantized_model(&quantized_path_str, &format_str, &output_path_str) {
        Ok(_) => TrustformersError::Success,
        Err(err) => err,
    }
}

/// Destroy quantization engine and free resources
#[no_mangle]
pub extern "C" fn trustformers_quantization_destroy_engine(
    handle: QuantizationHandle,
) -> TrustformersError {
    let mut engines = QUANTIZATION_ENGINES.write();
    match engines.remove(&handle) {
        Some(_) => TrustformersError::Success,
        None => TrustformersError::InvalidHandle,
    }
}

/// Get recommended quantization configuration
#[no_mangle]
pub extern "C" fn trustformers_quantization_recommend_config(
    model_size_mb: c_double,
    target_platform: *const c_char,
    accuracy_target: c_double,
    config: *mut TrustformersQuantizationConfig,
) -> TrustformersError {
    if target_platform.is_null() || config.is_null() {
        return TrustformersError::NullPointer;
    }

    let platform_str = match c_str_to_string(target_platform) {
        Ok(platform) => platform,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let recommended_config = super::utils::QuantizationUtils::recommend_quantization_config(
        model_size_mb,
        &platform_str,
        accuracy_target,
    );

    unsafe {
        (*config).quantization_type = recommended_config.quantization_type as c_int;
        (*config).weight_bits = recommended_config.weight_precision as c_int;
        (*config).activation_bits = recommended_config.activation_precision as c_int;
        (*config).quantize_weights = if recommended_config.quantize_weights { 1 } else { 0 };
        (*config).quantize_activations =
            if recommended_config.quantize_activations { 1 } else { 0 };
        (*config).calibration_method = recommended_config.calibration_method as c_int;
        (*config).calibration_samples = recommended_config.calibration_samples as c_int;
        (*config).symmetric_quantization =
            if recommended_config.range_settings.symmetric { 1 } else { 0 };
        (*config).enable_advanced_optimizations = 1;
        (*config).num_threads = recommended_config.performance_settings.num_threads as c_int;
        (*config).calibration_dataset_path = ptr::null();
    }

    TrustformersError::Success
}

/// Validate quantization configuration
#[no_mangle]
pub extern "C" fn trustformers_quantization_validate_config(
    config: *const TrustformersQuantizationConfig,
    warnings: *mut *mut c_char,
    num_warnings: *mut usize,
) -> TrustformersError {
    if config.is_null() {
        return TrustformersError::NullPointer;
    }

    // Convert C config to Rust config (simplified for validation)
    let c_config = unsafe { &*config };

    let quantization_type = match c_config.quantization_type {
        0 => QuantizationType::None,
        1 => QuantizationType::INT8,
        2 => QuantizationType::INT4,
        3 => QuantizationType::Dynamic,
        4 => QuantizationType::MixedPrecision,
        5 => QuantizationType::QAT,
        6 => QuantizationType::GPTQ,
        7 => QuantizationType::AWQ,
        8 => QuantizationType::SmoothQuant,
        9 => QuantizationType::GGML,
        99 => QuantizationType::Custom,
        _ => return TrustformersError::InvalidParameter,
    };

    let rust_config = QuantizationConfig {
        quantization_type,
        calibration_samples: c_config.calibration_samples as u32,
        ..Default::default()
    };

    match super::utils::QuantizationUtils::validate_config(&rust_config) {
        Ok(warning_list) => {
            if !warnings.is_null() && !num_warnings.is_null() {
                // Convert warnings to C strings (simplified implementation)
                unsafe {
                    *num_warnings = warning_list.len();
                    // Note: In a real implementation, you'd need to properly allocate
                    // and manage the C string array
                }
            }
            TrustformersError::Success
        },
        Err(err) => err,
    }
}

/// Get version information
#[no_mangle]
pub extern "C" fn trustformers_quantization_get_version(
    version: *mut *mut c_char,
) -> TrustformersError {
    if version.is_null() {
        return TrustformersError::NullPointer;
    }

    let version_str = env!("CARGO_PKG_VERSION");
    let c_str = str_to_c_str(version_str);
    if c_str.is_null() {
        TrustformersError::OutOfMemory
    } else {
        unsafe {
            *version = c_str;
        }
        TrustformersError::Success
    }
}
