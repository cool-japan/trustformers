//! C API for TrustformeRS Mobile
//!
//! This module provides C-compatible exports for the TrustformeRS mobile library,
//! enabling interoperability with languages like C#, C++, and other FFI consumers.

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use serde_json;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::{
    device_info::MobileDeviceDetector, inference::MobileInferenceEngine, MemoryOptimization,
    MobileBackend, MobileConfig, MobilePlatform,
};

// Error handling
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustformersMobileError {
    Success = 0,
    InvalidParameter = 1,
    OutOfMemory = 2,
    ModelLoadError = 3,
    InferenceError = 4,
    ConfigurationError = 5,
    PlatformNotSupported = 6,
    RuntimeError = 7,
    NullPointer = 8,
    SerializationError = 9,
}

// Global engine registry for C API
static ENGINE_REGISTRY: Lazy<Mutex<HashMap<usize, Arc<Mutex<MobileInferenceEngine>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Global counter for engine handles
static ENGINE_HANDLE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| {
    Mutex::new(5000) // Start at 5000 to avoid confusion with other handles
});

// Global config registry
static CONFIG_REGISTRY: Lazy<Mutex<HashMap<usize, MobileConfig>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Global counter for config handles
static CONFIG_HANDLE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| {
    Mutex::new(6000) // Start at 6000
});

// Helper functions
fn c_str_to_string(c_str: *const c_char) -> Result<String> {
    if c_str.is_null() {
        return Err(anyhow!("Null pointer"));
    }
    let c_str = unsafe { CStr::from_ptr(c_str) };
    Ok(c_str.to_string_lossy().into_owned())
}

fn string_to_c_str(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

// C API Functions

/// Initialize the TrustformeRS Mobile library
#[no_mangle]
pub extern "C" fn trustformers_mobile_init() -> TrustformersMobileError {
    // Initialize any global state if needed
    TrustformersMobileError::Success
}

/// Create a new mobile configuration with default settings
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_config_create_default(
    config_handle: *mut usize,
) -> TrustformersMobileError {
    if config_handle.is_null() {
        return TrustformersMobileError::NullPointer;
    }

    let config = MobileConfig::default();

    // Generate handle and store config
    let handle = match CONFIG_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    match CONFIG_REGISTRY.lock() {
        Ok(mut registry) => {
            registry.insert(handle, config);
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    }

    unsafe {
        *config_handle = handle;
    }

    TrustformersMobileError::Success
}

/// Create iOS optimized configuration
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_config_create_ios_optimized(
    config_handle: *mut usize,
) -> TrustformersMobileError {
    if config_handle.is_null() {
        return TrustformersMobileError::NullPointer;
    }

    let config = MobileConfig::ios_optimized();

    let handle = match CONFIG_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    match CONFIG_REGISTRY.lock() {
        Ok(mut registry) => {
            registry.insert(handle, config);
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    }

    unsafe {
        *config_handle = handle;
    }

    TrustformersMobileError::Success
}

/// Create Android optimized configuration
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_config_create_android_optimized(
    config_handle: *mut usize,
) -> TrustformersMobileError {
    if config_handle.is_null() {
        return TrustformersMobileError::NullPointer;
    }

    let config = MobileConfig::android_optimized();

    let handle = match CONFIG_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    match CONFIG_REGISTRY.lock() {
        Ok(mut registry) => {
            registry.insert(handle, config);
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    }

    unsafe {
        *config_handle = handle;
    }

    TrustformersMobileError::Success
}

/// Create ultra low memory configuration
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_config_create_ultra_low_memory(
    config_handle: *mut usize,
) -> TrustformersMobileError {
    if config_handle.is_null() {
        return TrustformersMobileError::NullPointer;
    }

    let config = MobileConfig::ultra_low_memory();

    let handle = match CONFIG_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    match CONFIG_REGISTRY.lock() {
        Ok(mut registry) => {
            registry.insert(handle, config);
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    }

    unsafe {
        *config_handle = handle;
    }

    TrustformersMobileError::Success
}

/// Set platform for configuration
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_set_platform(
    config_handle: usize,
    platform: c_int, // 0=iOS, 1=Android, 2=Generic
) -> TrustformersMobileError {
    let mut registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let config = match registry.get_mut(&config_handle) {
        Some(config) => config,
        None => return TrustformersMobileError::InvalidParameter,
    };

    config.platform = match platform {
        0 => MobilePlatform::Ios,
        1 => MobilePlatform::Android,
        2 => MobilePlatform::Generic,
        _ => return TrustformersMobileError::InvalidParameter,
    };

    TrustformersMobileError::Success
}

/// Set backend for configuration
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_set_backend(
    config_handle: usize,
    backend: c_int, // 0=CPU, 1=CoreML, 2=NNAPI, 3=GPU, 4=Metal, 5=Vulkan, 6=OpenCL, 7=Custom
) -> TrustformersMobileError {
    let mut registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let config = match registry.get_mut(&config_handle) {
        Some(config) => config,
        None => return TrustformersMobileError::InvalidParameter,
    };

    config.backend = match backend {
        0 => MobileBackend::CPU,
        1 => MobileBackend::CoreML,
        2 => MobileBackend::NNAPI,
        3 => MobileBackend::GPU,
        4 => MobileBackend::Metal,
        5 => MobileBackend::Vulkan,
        6 => MobileBackend::OpenCL,
        7 => MobileBackend::Custom,
        _ => return TrustformersMobileError::InvalidParameter,
    };

    TrustformersMobileError::Success
}

/// Set memory optimization level
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_set_memory_optimization(
    config_handle: usize,
    optimization: c_int, // 0=Minimal, 1=Balanced, 2=Maximum
) -> TrustformersMobileError {
    let mut registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let config = match registry.get_mut(&config_handle) {
        Some(config) => config,
        None => return TrustformersMobileError::InvalidParameter,
    };

    config.memory_optimization = match optimization {
        0 => MemoryOptimization::Minimal,
        1 => MemoryOptimization::Balanced,
        2 => MemoryOptimization::Maximum,
        _ => return TrustformersMobileError::InvalidParameter,
    };

    TrustformersMobileError::Success
}

/// Set maximum memory usage in MB
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_set_max_memory_mb(
    config_handle: usize,
    max_memory_mb: usize,
) -> TrustformersMobileError {
    let mut registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let config = match registry.get_mut(&config_handle) {
        Some(config) => config,
        None => return TrustformersMobileError::InvalidParameter,
    };

    config.max_memory_mb = max_memory_mb;
    TrustformersMobileError::Success
}

/// Enable or disable FP16 precision
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_set_use_fp16(
    config_handle: usize,
    use_fp16: c_int, // 0=false, 1=true
) -> TrustformersMobileError {
    let mut registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let config = match registry.get_mut(&config_handle) {
        Some(config) => config,
        None => return TrustformersMobileError::InvalidParameter,
    };

    config.use_fp16 = use_fp16 != 0;
    TrustformersMobileError::Success
}

/// Set thread count
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_set_num_threads(
    config_handle: usize,
    num_threads: usize,
) -> TrustformersMobileError {
    let mut registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let config = match registry.get_mut(&config_handle) {
        Some(config) => config,
        None => return TrustformersMobileError::InvalidParameter,
    };

    config.num_threads = num_threads;
    TrustformersMobileError::Success
}

/// Validate configuration
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_validate(
    config_handle: usize,
) -> TrustformersMobileError {
    let registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let config = match registry.get(&config_handle) {
        Some(config) => config,
        None => return TrustformersMobileError::InvalidParameter,
    };

    match config.validate() {
        Ok(_) => TrustformersMobileError::Success,
        Err(_) => TrustformersMobileError::ConfigurationError,
    }
}

/// Free configuration handle
#[no_mangle]
pub extern "C" fn trustformers_mobile_config_free(config_handle: usize) -> TrustformersMobileError {
    let mut registry = match CONFIG_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    if registry.remove(&config_handle).is_some() {
        TrustformersMobileError::Success
    } else {
        TrustformersMobileError::InvalidParameter
    }
}

/// Create inference engine with configuration
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_engine_create(
    config_handle: usize,
    model_path: *const c_char,
    engine_handle: *mut usize,
) -> TrustformersMobileError {
    if model_path.is_null() || engine_handle.is_null() {
        return TrustformersMobileError::NullPointer;
    }

    let model_path_str = match c_str_to_string(model_path) {
        Ok(s) => s,
        Err(_) => return TrustformersMobileError::InvalidParameter,
    };

    // Get configuration
    let config = {
        let registry = match CONFIG_REGISTRY.lock() {
            Ok(registry) => registry,
            Err(_) => return TrustformersMobileError::RuntimeError,
        };

        match registry.get(&config_handle) {
            Some(config) => config.clone(),
            None => return TrustformersMobileError::InvalidParameter,
        }
    };

    // Create engine
    let mut engine = match MobileInferenceEngine::new(config) {
        Ok(engine) => engine,
        Err(_) => return TrustformersMobileError::ModelLoadError,
    };

    // Load model
    if engine.load_model_from_file(&model_path_str).is_err() {
        return TrustformersMobileError::ModelLoadError;
    }

    // Generate handle and store engine
    let handle = match ENGINE_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    match ENGINE_REGISTRY.lock() {
        Ok(mut registry) => {
            registry.insert(handle, Arc::new(Mutex::new(engine)));
        },
        Err(_) => return TrustformersMobileError::RuntimeError,
    }

    unsafe {
        *engine_handle = handle;
    }

    TrustformersMobileError::Success
}

/// Run inference with float32 input
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_engine_inference_f32(
    engine_handle: usize,
    input_data: *const c_float,
    input_size: usize,
    output_data: *mut c_float,
    output_size: usize,
    actual_output_size: *mut usize,
) -> TrustformersMobileError {
    if input_data.is_null() || output_data.is_null() || actual_output_size.is_null() {
        return TrustformersMobileError::NullPointer;
    }

    let registry = match ENGINE_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let engine_arc = match registry.get(&engine_handle) {
        Some(engine) => engine.clone(),
        None => return TrustformersMobileError::InvalidParameter,
    };

    drop(registry); // Release lock

    let input_slice = unsafe { std::slice::from_raw_parts(input_data, input_size) };

    let output_slice = unsafe { std::slice::from_raw_parts_mut(output_data, output_size) };

    let mut engine = match engine_arc.lock() {
        Ok(engine) => engine,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    match engine.inference_f32(input_slice, output_slice) {
        Ok(size) => {
            unsafe {
                *actual_output_size = size;
            }
            TrustformersMobileError::Success
        },
        Err(_) => TrustformersMobileError::InferenceError,
    }
}

/// Get device information as JSON string
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_get_device_info(
    device_info_json: *mut *mut c_char,
) -> TrustformersMobileError {
    if device_info_json.is_null() {
        return TrustformersMobileError::NullPointer;
    }

    let device_info = match MobileDeviceDetector::detect() {
        Ok(info) => info,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    let json_string = match serde_json::to_string_pretty(&device_info) {
        Ok(json) => json,
        Err(_) => return TrustformersMobileError::SerializationError,
    };

    unsafe {
        *device_info_json = string_to_c_str(json_string);
    }

    TrustformersMobileError::Success
}

/// Free engine handle
#[no_mangle]
pub extern "C" fn trustformers_mobile_engine_free(engine_handle: usize) -> TrustformersMobileError {
    let mut registry = match ENGINE_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersMobileError::RuntimeError,
    };

    if registry.remove(&engine_handle).is_some() {
        TrustformersMobileError::Success
    } else {
        TrustformersMobileError::InvalidParameter
    }
}

/// Free C string allocated by the library
#[no_mangle]
pub unsafe extern "C" fn trustformers_mobile_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

/// Get library version
#[no_mangle]
pub extern "C" fn trustformers_mobile_version() -> *const c_char {
    b"1.0.0\0".as_ptr() as *const c_char
}

/// Check platform support
#[no_mangle]
pub extern "C" fn trustformers_mobile_is_platform_supported(platform: c_int) -> c_int {
    match platform {
        0 => {
            // iOS
            #[cfg(target_os = "ios")]
            return 1;
            #[cfg(not(target_os = "ios"))]
            return 0;
        },
        1 => {
            // Android
            #[cfg(target_os = "android")]
            return 1;
            #[cfg(not(target_os = "android"))]
            return 0;
        },
        2 => 1, // Generic - always supported
        _ => 0,
    }
}

/// Check backend support
#[no_mangle]
pub extern "C" fn trustformers_mobile_is_backend_supported(backend: c_int) -> c_int {
    match backend {
        0 => 1, // CPU - always supported
        1 => {
            // CoreML
            #[cfg(feature = "coreml")]
            return 1;
            #[cfg(not(feature = "coreml"))]
            return 0;
        },
        2 => {
            // NNAPI
            #[cfg(feature = "nnapi")]
            return 1;
            #[cfg(not(feature = "nnapi"))]
            return 0;
        },
        3 => 1, // GPU - basic support always available
        4 => {
            // Metal
            #[cfg(target_os = "ios")]
            return 1;
            #[cfg(not(target_os = "ios"))]
            return 0;
        },
        5 => 1, // Vulkan - cross-platform support
        6 => 1, // OpenCL - cross-platform support
        7 => 1, // Custom - always supported
        _ => 0,
    }
}
