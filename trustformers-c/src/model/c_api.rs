//! C API functions for model operations

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int};
use std::ptr;

use crate::core_result_to_error;
use crate::core_tensor_result_to_error;
use crate::error::*;
use crate::trustformers_result_to_error;
use crate::{c_str_to_string, result_to_error, string_to_c_str, RESOURCE_REGISTRY};

use trustformers::{AutoModel, Model};
use trustformers_core::{tensor::Tensor, Config};

use super::types::*;

/// Helper struct for validation results
#[derive(Debug)]
struct ValidationResult {
    is_valid: bool,
    errors: Vec<String>,
    warnings: Vec<String>,
    checksum_valid: bool,
    file_integrity_valid: bool,
    config_valid: bool,
}

/// Load a model from a pretrained checkpoint
#[no_mangle]
pub extern "C" fn trustformers_model_from_pretrained(
    config: *const TrustformersModelConfig,
    model_handle: *mut TrustformersModel,
) -> TrustformersError {
    if config.is_null() || model_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let config = unsafe { &*config };

    let model_name = c_try!(c_str_to_string(config.model_name));

    // Load model
    let model_result = AutoModel::from_pretrained(&model_name);
    let (error, model_opt) = trustformers_result_to_error(model_result);

    if error != TrustformersError::Success {
        return error;
    }

    let model = model_opt.unwrap();

    // Register model and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_model(model);

    unsafe {
        *model_handle = handle;
    }

    TrustformersError::Success
}

/// Load a model from a local path
#[no_mangle]
pub extern "C" fn trustformers_model_from_path(
    path: *const c_char,
    model_handle: *mut TrustformersModel,
) -> TrustformersError {
    if path.is_null() || model_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = c_try!(c_str_to_string(path));

    // Load model from path
    let model_result = AutoModel::from_pretrained(&path_str);
    let (error, model_opt) = trustformers_result_to_error(model_result);

    if error != TrustformersError::Success {
        return error;
    }

    let model = model_opt.unwrap();

    // Register model and return handle
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_model(model);

    unsafe {
        *model_handle = handle;
    }

    TrustformersError::Success
}

/// Perform forward pass through the model
#[no_mangle]
pub extern "C" fn trustformers_model_forward(
    model_handle: TrustformersModel,
    inputs: *const TrustformersTensor,
    num_inputs: usize,
    outputs: *mut *mut TrustformersTensor,
    num_outputs: *mut usize,
) -> TrustformersError {
    if inputs.is_null() || outputs.is_null() || num_outputs.is_null() || num_inputs == 0 {
        return TrustformersError::InvalidParameter;
    }

    // Get model from registry
    let registry = RESOURCE_REGISTRY.read();
    let model_arc = match registry.get_model(model_handle) {
        Some(m) => m,
        None => return TrustformersError::InvalidParameter,
    };

    // Convert input tensor handles to actual tensors
    let mut input_tensors = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        unsafe {
            let tensor_handle = *inputs.add(i);
            if let Some(tensor_arc) = registry.get_tensor(tensor_handle) {
                // tensor_arc is &Arc<Tensor>, so we can clone the Arc and get the inner value
                input_tensors.push((**tensor_arc).clone());
            } else {
                return TrustformersError::InvalidParameter;
            }
        }
    }

    // Perform forward pass (this is a simplified example)
    // model_arc is &Arc<AutoModel>, which implements Model trait
    {
        // In a real implementation, you might need to concatenate or process multiple inputs
        let first_input = &input_tensors[0];

        let result = model_arc.forward(first_input.clone());
        let (error, result_opt) = core_tensor_result_to_error(result);

        if error != TrustformersError::Success {
            return error;
        }

        let output_tensor = result_opt.unwrap();

        // Register output tensor(s)
        drop(registry);
        let mut registry = RESOURCE_REGISTRY.write();
        let output_handle = registry.register_tensor(output_tensor);

        // For simplicity, we're returning just one output tensor
        unsafe {
            let output_handles =
                libc::malloc(std::mem::size_of::<TrustformersTensor>()) as *mut TrustformersTensor;
            *output_handles = output_handle;
            *outputs = output_handles;
            *num_outputs = 1;
        }

        TrustformersError::Success
    }
}

/// Get model metadata
#[no_mangle]
pub extern "C" fn trustformers_model_metadata(
    model_handle: TrustformersModel,
    metadata: *mut TrustformersModelMetadata,
) -> TrustformersError {
    if metadata.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = RESOURCE_REGISTRY.read();
    let model_arc = match registry.get_model(model_handle) {
        Some(m) => m,
        None => return TrustformersError::InvalidParameter,
    };

    // Get model config/metadata (simplified)
    unsafe {
        (*metadata).architecture = string_to_c_str("transformer".to_string());
        (*metadata).num_parameters = 7_000_000_000; // Example: 7B parameters
        (*metadata).model_size_bytes = 13_000_000_000; // Example: 13GB
        (*metadata).vocab_size = 32000;
        (*metadata).hidden_size = 4096;
        (*metadata).num_layers = 32;
        (*metadata).num_attention_heads = 32;
        (*metadata).max_sequence_length = 2048;
        (*metadata).model_format = string_to_c_str("safetensors".to_string());
        (*metadata).framework = string_to_c_str("trustformers".to_string());
        (*metadata).model_version = string_to_c_str("1.0.0".to_string());
        (*metadata).license = string_to_c_str("MIT".to_string());
        (*metadata).is_quantized = 0; // False
        (*metadata).quantization_info = ptr::null_mut();
    }

    TrustformersError::Success
}

/// Create tensor from data
#[no_mangle]
pub extern "C" fn trustformers_tensor_from_data(
    data: *const c_float,
    shape: *const usize,
    ndim: usize,
    tensor_handle: *mut TrustformersTensor,
) -> TrustformersError {
    if data.is_null() || shape.is_null() || tensor_handle.is_null() || ndim == 0 {
        return TrustformersError::NullPointer;
    }

    // Convert shape from C array to Vec
    let shape_vec: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, ndim).to_vec() };

    // Calculate number of elements
    let num_elements: usize = shape_vec.iter().product();

    // Convert data from C array to Vec
    let data_vec: Vec<f32> = unsafe { std::slice::from_raw_parts(data, num_elements).to_vec() };

    // Create tensor
    let tensor_result = Tensor::from_vec(data_vec, &shape_vec);
    let (error, tensor_opt) = core_tensor_result_to_error(tensor_result);

    if error != TrustformersError::Success {
        return error;
    }

    let tensor = tensor_opt.unwrap();

    // Register tensor
    let mut registry = RESOURCE_REGISTRY.write();
    let handle = registry.register_tensor(tensor);

    unsafe {
        *tensor_handle = handle;
    }

    TrustformersError::Success
}

/// Get tensor information
#[no_mangle]
pub extern "C" fn trustformers_tensor_info(
    tensor_handle: TrustformersTensor,
    info: *mut TrustformersTensorInfo,
) -> TrustformersError {
    if info.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = RESOURCE_REGISTRY.read();
    let tensor_arc = match registry.get_tensor(tensor_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tensor_arc is &Arc<Tensor>, get reference to inner Tensor
    let tensor = tensor_arc.as_ref();
    let shape = tensor.shape();

    unsafe {
        // Allocate memory for shape array
        let shape_ptr = libc::malloc(shape.len() * std::mem::size_of::<usize>()) as *mut usize;
        for (i, &dim) in shape.iter().enumerate() {
            *shape_ptr.add(i) = dim;
        }

        (*info).shape = shape_ptr;
        (*info).ndim = shape.len();
        (*info).dtype = 0; // float32 (simplified)
        (*info).numel = tensor.size();
        (*info).size_bytes = tensor.size() * std::mem::size_of::<f32>();
        (*info).device_type = 0; // CPU (simplified)
        (*info).device_id = 0;
    }

    TrustformersError::Success
}

/// Get tensor data
#[no_mangle]
pub extern "C" fn trustformers_tensor_data(
    tensor_handle: TrustformersTensor,
    data: *mut *mut c_float,
    num_elements: *mut usize,
) -> TrustformersError {
    if data.is_null() || num_elements.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = RESOURCE_REGISTRY.read();
    let tensor_arc = match registry.get_tensor(tensor_handle) {
        Some(t) => t,
        None => return TrustformersError::InvalidParameter,
    };

    // tensor_arc is &Arc<Tensor>, get reference to inner Tensor
    let tensor = tensor_arc.as_ref();
    let tensor_data = tensor.to_vec_f32().unwrap_or_default();

    unsafe {
        // Allocate memory for data
        let data_ptr = libc::malloc(tensor_data.len() * std::mem::size_of::<f32>()) as *mut c_float;
        for (i, &val) in tensor_data.iter().enumerate() {
            *data_ptr.add(i) = val;
        }

        *data = data_ptr;
        *num_elements = tensor_data.len();
    }

    TrustformersError::Success
}

/// Free tensor info
#[no_mangle]
pub extern "C" fn trustformers_tensor_info_free(info: *mut TrustformersTensorInfo) {
    if !info.is_null() {
        unsafe {
            let tensor_info = &mut *info;
            if !tensor_info.shape.is_null() {
                libc::free(tensor_info.shape as *mut std::ffi::c_void);
                tensor_info.shape = ptr::null_mut();
            }
        }
    }
}

/// Free tensor data
#[no_mangle]
pub extern "C" fn trustformers_tensor_data_free(data: *mut c_float, num_elements: usize) {
    if !data.is_null() && num_elements > 0 {
        unsafe {
            libc::free(data as *mut std::ffi::c_void);
        }
    }
}

/// Free model metadata
#[no_mangle]
pub extern "C" fn trustformers_model_metadata_free(metadata: *mut TrustformersModelMetadata) {
    if !metadata.is_null() {
        unsafe {
            let meta = &mut *metadata;
            if !meta.architecture.is_null() {
                let _ = CString::from_raw(meta.architecture);
            }
            if !meta.model_format.is_null() {
                let _ = CString::from_raw(meta.model_format);
            }
            if !meta.framework.is_null() {
                let _ = CString::from_raw(meta.framework);
            }
            if !meta.model_version.is_null() {
                let _ = CString::from_raw(meta.model_version);
            }
            if !meta.license.is_null() {
                let _ = CString::from_raw(meta.license);
            }
            if !meta.quantization_info.is_null() {
                let _ = CString::from_raw(meta.quantization_info);
            }
        }
    }
}

/// Destroy model
#[no_mangle]
pub extern "C" fn trustformers_model_destroy(model_handle: TrustformersModel) -> TrustformersError {
    let mut registry = RESOURCE_REGISTRY.write();
    match registry.unregister_model(model_handle) {
        true => TrustformersError::Success,
        false => TrustformersError::InvalidParameter,
    }
}

/// Destroy tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_destroy(
    tensor_handle: TrustformersTensor,
) -> TrustformersError {
    let mut registry = RESOURCE_REGISTRY.write();
    match registry.unregister_tensor(tensor_handle) {
        true => TrustformersError::Success,
        false => TrustformersError::InvalidParameter,
    }
}

/// Validate model
#[no_mangle]
pub extern "C" fn trustformers_model_validate(
    path: *const c_char,
    validation: *mut TrustformersModelValidation,
) -> TrustformersError {
    if path.is_null() || validation.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = c_try!(c_str_to_string(path));

    // Perform validation (simplified)
    let validation_result = validate_model_path(&path_str);

    unsafe {
        (*validation).is_valid = if validation_result.is_valid { 1 } else { 0 };
        (*validation).checksum_valid = if validation_result.checksum_valid { 1 } else { 0 };
        (*validation).file_integrity_valid =
            if validation_result.file_integrity_valid { 1 } else { 0 };
        (*validation).config_valid = if validation_result.config_valid { 1 } else { 0 };

        let errors_json = serde_json::to_string(&validation_result.errors).unwrap_or_default();
        let warnings_json = serde_json::to_string(&validation_result.warnings).unwrap_or_default();

        (*validation).errors = string_to_c_str(errors_json);
        (*validation).warnings = string_to_c_str(warnings_json);
    }

    TrustformersError::Success
}

/// Detect model format
#[no_mangle]
pub extern "C" fn trustformers_model_detect_format(
    path: *const c_char,
    format: *mut *mut c_char,
) -> TrustformersError {
    if path.is_null() || format.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = c_try!(c_str_to_string(path));

    // Simple format detection based on file extension
    let detected_format = if path_str.contains(".safetensors") {
        "safetensors"
    } else if path_str.contains(".bin") || path_str.contains(".pt") || path_str.contains(".pth") {
        "pytorch"
    } else if path_str.contains(".onnx") {
        "onnx"
    } else if path_str.contains(".pb") {
        "tensorflow"
    } else if path_str.contains(".ggml") || path_str.contains(".gguf") {
        "ggml"
    } else {
        "unknown"
    };

    unsafe {
        *format = string_to_c_str(detected_format.to_string());
    }

    TrustformersError::Success
}

/// Check if model format is supported
#[no_mangle]
pub extern "C" fn trustformers_model_format_supported(format: *const c_char) -> c_int {
    if format.is_null() {
        return 0; // False
    }

    let format_str = match c_str_to_string(format) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let supported_formats = ["pytorch", "safetensors", "onnx", "tensorflow", "ggml"];
    if supported_formats.contains(&format_str.as_str()) {
        1 // True
    } else {
        0 // False
    }
}

/// Free validation result
#[no_mangle]
pub extern "C" fn trustformers_model_validation_free(validation: *mut TrustformersModelValidation) {
    if !validation.is_null() {
        unsafe {
            let val = &mut *validation;
            if !val.errors.is_null() {
                let _ = CString::from_raw(val.errors);
            }
            if !val.warnings.is_null() {
                let _ = CString::from_raw(val.warnings);
            }
        }
    }
}

// Helper function for model validation
fn validate_model_path(path: &str) -> ValidationResult {
    let mut result = ValidationResult {
        is_valid: false,
        errors: Vec::new(),
        warnings: Vec::new(),
        checksum_valid: false,
        file_integrity_valid: false,
        config_valid: false,
    };

    // Check for required files (simplified validation)
    if std::path::Path::new(path).exists() {
        result.file_integrity_valid = true;
        result.checksum_valid = true; // Simplified - would check actual checksums
        result.config_valid = true; // Simplified - would validate config
        result.is_valid = true;
    } else {
        result.errors.push("Path does not exist".to_string());
    }

    result
}
