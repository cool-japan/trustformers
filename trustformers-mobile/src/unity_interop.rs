//! Unity C# interoperability layer for TrustformeRS mobile
//!
//! This module provides C-compatible FFI functions that can be called from Unity's C# scripts.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

use serde_json;
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

use crate::{
    device_info::MobileDeviceDetector, inference::MobileInferenceEngine, MobileConfig, MobileStats,
};

// Engine instance storage
use std::sync::OnceLock;

static ENGINE_STORAGE: OnceLock<Mutex<HashMap<usize, Arc<Mutex<MobileInferenceEngine>>>>> =
    OnceLock::new();
static NEXT_ENGINE_ID: OnceLock<Mutex<usize>> = OnceLock::new();

fn get_engine_storage() -> &'static Mutex<HashMap<usize, Arc<Mutex<MobileInferenceEngine>>>> {
    ENGINE_STORAGE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_next_engine_id() -> &'static Mutex<usize> {
    NEXT_ENGINE_ID.get_or_init(|| Mutex::new(1))
}

// Callback function types for IL2CPP compatibility
type LogCallback = extern "C" fn(*const c_char);
type ProgressCallback = extern "C" fn(c_float);
type ErrorCallback = extern "C" fn(c_int, *const c_char);
type InferenceCompleteCallback = extern "C" fn(*const c_float, c_int);

// Global callback storage
static mut LOG_CALLBACK: Option<LogCallback> = None;
static mut PROGRESS_CALLBACK: Option<ProgressCallback> = None;
static mut ERROR_CALLBACK: Option<ErrorCallback> = None;
static mut INFERENCE_CALLBACK: Option<InferenceCompleteCallback> = None;

// Helper macro for C string creation
macro_rules! c_string {
    ($s:expr) => {
        CString::new($s)
            .unwrap_or_else(|_| CString::new("Invalid string").expect("Operation failed"))
            .into_raw()
    };
}

// Helper macro for error handling with callbacks
macro_rules! handle_error {
    ($result:expr, $default:expr) => {
        match $result {
            Ok(val) => val,
            Err(e) => {
                let error_msg = format!("TrustformeRS Error: {}", e);
                let c_msg = CString::new(error_msg).expect("Operation failed");
                unsafe {
                    if let Some(callback) = ERROR_CALLBACK {
                        callback(-1, c_msg.as_ptr());
                    }
                }
                return $default;
            },
        }
    };
}

/// Initialize IL2CPP support
#[no_mangle]
pub extern "C" fn trustformers_initialize_il2cpp_support() -> c_int {
    log::info("Initializing IL2CPP support for TrustformeRS");

    // Initialize logging (Unity-specific logging is handled via callbacks)
    // No additional initialization needed for Unity logging

    0 // Success
}

/// Cleanup IL2CPP support
#[no_mangle]
pub extern "C" fn trustformers_cleanup_il2cpp_support() {
    log::info("Cleaning up IL2CPP support");

    // Clear all engine instances
    if let Ok(mut storage) = get_engine_storage().lock() {
        storage.clear();
    }

    // Clear callbacks
    unsafe {
        LOG_CALLBACK = None;
        PROGRESS_CALLBACK = None;
        ERROR_CALLBACK = None;
        INFERENCE_CALLBACK = None;
    }
}

/// Set log callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_log_callback(callback: LogCallback) {
    unsafe {
        LOG_CALLBACK = Some(callback);
    }
}

/// Set progress callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_progress_callback(callback: ProgressCallback) {
    unsafe {
        PROGRESS_CALLBACK = Some(callback);
    }
}

/// Set error callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_error_callback(callback: ErrorCallback) {
    unsafe {
        ERROR_CALLBACK = Some(callback);
    }
}

/// Set inference complete callback for IL2CPP
#[no_mangle]
pub extern "C" fn trustformers_set_inference_callback(callback: InferenceCompleteCallback) {
    unsafe {
        INFERENCE_CALLBACK = Some(callback);
    }
}

/// Create a new TrustformeRS engine
#[no_mangle]
pub extern "C" fn trustformers_create_engine(config_json: *const c_char) -> *mut c_void {
    if config_json.is_null() {
        return ptr::null_mut();
    }

    let config_str = unsafe {
        match CStr::from_ptr(config_json).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let config: MobileConfig = handle_error!(
        serde_json::from_str(config_str).map_err(|e| {
            trustformers_core::error::CoreError::from(TrustformersError::config_error(
                &e.to_string(),
                "parse_config",
            ))
        }),
        ptr::null_mut()
    );

    tracing::info!("Creating TrustformeRS engine with config: {:?}", config);

    let engine = handle_error!(MobileInferenceEngine::new(config), ptr::null_mut());

    // Store engine and return ID as pointer
    let engine_arc = Arc::new(Mutex::new(engine));
    let mut storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        ptr::null_mut()
    );
    let mut next_id = handle_error!(
        get_next_engine_id()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        ptr::null_mut()
    );

    let engine_id = *next_id;
    *next_id += 1;

    storage.insert(engine_id, engine_arc);

    tracing::info!("Engine created with ID: {}", engine_id);
    engine_id as *mut c_void
}

/// Destroy a TrustformeRS engine
#[no_mangle]
pub extern "C" fn trustformers_destroy_engine(engine_ptr: *mut c_void) {
    if engine_ptr.is_null() {
        return;
    }

    let engine_id = engine_ptr as usize;

    if let Ok(mut storage) = get_engine_storage().lock() {
        if storage.remove(&engine_id).is_some() {
            log::info(&format!("Engine {} destroyed", engine_id));
        }
    }
}

/// Load a model into the engine
#[no_mangle]
pub extern "C" fn trustformers_load_model(
    engine_ptr: *mut c_void,
    model_path: *const c_char,
) -> c_int {
    if engine_ptr.is_null() || model_path.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;
    let path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    tracing::info!("Loading model: {}", path_str);

    // Report progress
    unsafe {
        if let Some(callback) = PROGRESS_CALLBACK {
            callback(0.0);
        }
    }

    let result = engine.load_model_from_file(path_str);

    unsafe {
        if let Some(callback) = PROGRESS_CALLBACK {
            callback(1.0);
        }
    }

    match result {
        Ok(_) => {
            log::info(&format!("Model loaded successfully: {}", path_str));
            0
        },
        Err(e) => {
            let error_msg = format!("Failed to load model: {}", e);
            log::error(&error_msg);
            unsafe {
                if let Some(callback) = ERROR_CALLBACK {
                    let c_msg = CString::new(error_msg).expect("Operation failed");
                    callback(-2, c_msg.as_ptr());
                }
            }
            -2
        },
    }
}

/// Perform inference
#[no_mangle]
pub extern "C" fn trustformers_inference(
    engine_ptr: *mut c_void,
    input_data: *const c_float,
    input_length: c_int,
    output_data: *mut c_float,
    output_length: c_int,
) -> c_int {
    if engine_ptr.is_null() || input_data.is_null() || output_data.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;
    let input_slice = unsafe { std::slice::from_raw_parts(input_data, input_length as usize) };
    let output_slice =
        unsafe { std::slice::from_raw_parts_mut(output_data, output_length as usize) };

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Create input tensor
    let input_tensor = handle_error!(
        Tensor::from_slice(input_slice, &[input_length as usize]),
        -1
    );

    // Perform inference
    let result_tensor = handle_error!(engine.inference(&input_tensor), -1);

    // Copy results to output buffer
    let result_data = handle_error!(result_tensor.data(), -1);
    let copy_length = std::cmp::min(result_data.len(), output_slice.len());

    for i in 0..copy_length {
        output_slice[i] = result_data[i];
    }

    // Trigger async callback if available
    unsafe {
        if let Some(callback) = INFERENCE_CALLBACK {
            callback(output_data, copy_length as c_int);
        }
    }

    0 // Success
}

/// Perform batch inference
#[no_mangle]
pub extern "C" fn trustformers_batch_inference(
    engine_ptr: *mut c_void,
    input_data: *const c_float,
    batch_size: c_int,
    input_length: c_int,
    output_data: *mut c_float,
    output_length: c_int,
) -> c_int {
    if engine_ptr.is_null() || input_data.is_null() || output_data.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;
    let total_input_size = (batch_size * input_length) as usize;
    let total_output_size = (batch_size * output_length) as usize;

    let input_slice = unsafe { std::slice::from_raw_parts(input_data, total_input_size) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output_data, total_output_size) };

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Create batch input tensors
    let mut input_tensors = Vec::new();
    for i in 0..batch_size as usize {
        let start_idx = i * input_length as usize;
        let end_idx = start_idx + input_length as usize;
        let batch_input = &input_slice[start_idx..end_idx];

        let tensor = handle_error!(
            Tensor::from_slice(batch_input, &[input_length as usize]),
            -1
        );
        input_tensors.push(tensor);
    }

    // Perform batch inference
    let result_tensors = handle_error!(engine.batch_inference(input_tensors), -1);

    // Copy results to output buffer
    for (i, result_tensor) in result_tensors.iter().enumerate() {
        let result_data = handle_error!(result_tensor.data(), -1);
        let start_idx = i * output_length as usize;
        let copy_length = std::cmp::min(result_data.len(), output_length as usize);

        for j in 0..copy_length {
            if start_idx + j < output_slice.len() {
                output_slice[start_idx + j] = result_data[j];
            }
        }
    }

    0 // Success
}

/// Get engine statistics
#[no_mangle]
pub extern "C" fn trustformers_get_stats(engine_ptr: *mut c_void, stats: *mut MobileStats) {
    if engine_ptr.is_null() || stats.is_null() {
        return;
    }

    let engine_id = engine_ptr as usize;

    let storage = match get_engine_storage().lock() {
        Ok(s) => s,
        Err(_) => return,
    };

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return,
    };

    drop(storage); // Release lock early

    let engine = match engine_arc.lock() {
        Ok(e) => e,
        Err(_) => return,
    };

    let engine_stats = engine.get_stats();

    unsafe {
        *stats = engine_stats.clone();
    }
}

/// Set performance mode
#[no_mangle]
pub extern "C" fn trustformers_set_performance_mode(engine_ptr: *mut c_void, mode: c_int) -> c_int {
    if engine_ptr.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Set performance mode using the engine's method
    match engine.set_performance_mode(mode) {
        Ok(_) => {
            tracing::info!("Performance mode set to: {}", mode);
            0
        },
        Err(e) => {
            let error_msg = format!("Failed to set performance mode: {}", e);
            log::error(&error_msg);
            unsafe {
                if let Some(callback) = ERROR_CALLBACK {
                    let c_msg = CString::new(error_msg).expect("Operation failed");
                    callback(-3, c_msg.as_ptr());
                }
            }
            -3
        },
    }
}

/// Get device information
#[no_mangle]
pub extern "C" fn trustformers_get_device_info() -> *mut c_char {
    match MobileDeviceDetector::detect() {
        Ok(device_info) => {
            let info_json = match serde_json::to_string_pretty(&device_info) {
                Ok(json) => json,
                Err(e) => format!("Error serializing device info: {}", e),
            };
            c_string!(info_json)
        },
        Err(e) => {
            let error_msg = format!("Error getting device info: {}", e);
            c_string!(error_msg)
        },
    }
}

/// Free a string allocated by the native library
#[no_mangle]
pub extern "C" fn trustformers_free_string(str_ptr: *mut c_char) {
    if !str_ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(str_ptr);
        }
    }
}

/// Warm up the engine
#[no_mangle]
pub extern "C" fn trustformers_warm_up(engine_ptr: *mut c_void) -> c_int {
    if engine_ptr.is_null() {
        return -1;
    }

    let engine_id = engine_ptr as usize;

    let storage = handle_error!(
        get_engine_storage()
            .lock()
            .map_err(|_| trustformers_core::error::CoreError::from(
                TrustformersError::runtime_error("Lock poisoned".into())
            )),
        -1
    );

    let engine_arc = match storage.get(&engine_id) {
        Some(arc) => arc.clone(),
        None => return -1,
    };

    drop(storage); // Release lock early

    let mut engine = handle_error!(
        engine_arc.lock().map_err(|_| trustformers_core::error::CoreError::from(
            TrustformersError::runtime_error("Lock poisoned".into())
        )),
        -1
    );

    // Warm up the engine
    log::info("Starting engine warm-up");
    match engine.warm_up() {
        Ok(_) => {
            log::info("Engine warm-up completed successfully");
            0
        },
        Err(e) => {
            let error_msg = format!("Failed to warm up engine: {}", e);
            log::error(&error_msg);
            unsafe {
                if let Some(callback) = ERROR_CALLBACK {
                    let c_msg = CString::new(error_msg).expect("Operation failed");
                    callback(-3, c_msg.as_ptr());
                }
            }
            -3
        },
    }
}

// Memory management functions for IL2CPP compatibility

/// Allocate managed memory
#[no_mangle]
pub extern "C" fn trustformers_allocate_managed_memory(size: c_int) -> *mut c_void {
    if size <= 0 {
        return ptr::null_mut();
    }

    let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap_or_else(|_| {
        std::alloc::Layout::from_size_align(size as usize, 1).expect("Operation failed")
    });

    unsafe { std::alloc::alloc(layout) as *mut c_void }
}

/// Free managed memory
#[no_mangle]
pub extern "C" fn trustformers_free_managed_memory(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    // Note: In a real implementation, you'd need to store the original layout
    // This is simplified for demonstration
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, std::alloc::Layout::new::<u8>());
    }
}

/// Copy data to managed memory
#[no_mangle]
pub extern "C" fn trustformers_copy_to_managed_memory(
    dest: *mut c_void,
    source: *const c_float,
    length: c_int,
) {
    if dest.is_null() || source.is_null() || length <= 0 {
        return;
    }

    let src_slice = unsafe { std::slice::from_raw_parts(source, length as usize) };

    let dest_slice =
        unsafe { std::slice::from_raw_parts_mut(dest as *mut c_float, length as usize) };

    dest_slice.copy_from_slice(src_slice);
}

/// Copy data from managed memory
#[no_mangle]
pub extern "C" fn trustformers_copy_from_managed_memory(
    dest: *mut c_float,
    source: *const c_void,
    length: c_int,
) {
    if dest.is_null() || source.is_null() || length <= 0 {
        return;
    }

    let src_slice =
        unsafe { std::slice::from_raw_parts(source as *const c_float, length as usize) };

    let dest_slice = unsafe { std::slice::from_raw_parts_mut(dest, length as usize) };

    dest_slice.copy_from_slice(src_slice);
}

// Utility functions for logging
mod log {
    use super::*;

    pub fn info(message: &str) {
        let c_msg = CString::new(message).expect("Operation failed");
        unsafe {
            if let Some(callback) = LOG_CALLBACK {
                callback(c_msg.as_ptr());
            }
        }
        println!("[INFO] {}", message);
    }

    pub fn error(message: &str) {
        let c_msg = CString::new(message).expect("Operation failed");
        unsafe {
            if let Some(callback) = ERROR_CALLBACK {
                callback(-1, c_msg.as_ptr());
            }
        }
        eprintln!("[ERROR] {}", message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation_and_destruction() {
        let config = MobileConfig::default();
        let config_json = serde_json::to_string(&config).expect("Operation failed");
        let c_config = CString::new(config_json).expect("Operation failed");

        let engine_ptr = trustformers_create_engine(c_config.as_ptr());
        assert!(!engine_ptr.is_null());

        trustformers_destroy_engine(engine_ptr);
    }

    #[test]
    fn test_memory_management() {
        let size = 1024;
        let ptr = trustformers_allocate_managed_memory(size);
        assert!(!ptr.is_null());

        trustformers_free_managed_memory(ptr);
    }

    #[test]
    fn test_device_info() {
        let info_ptr = trustformers_get_device_info();
        assert!(!info_ptr.is_null());

        trustformers_free_string(info_ptr);
    }
}
