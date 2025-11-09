//! C API functions for ROCm backend

use std::os::raw::{c_char, c_float, c_int};

use super::operations::RocmOperations;
use super::types::*;
use super::{ROCM_MANAGER, ROCM_TENSOR_HANDLE_COUNTER, ROCM_TENSOR_REGISTRY};
use crate::error::TrustformersError;
use crate::{c_str_to_string, string_to_c_str};

/// Initialize ROCm backend
#[no_mangle]
pub extern "C" fn trustformers_rocm_init() -> TrustformersError {
    let mut manager = match ROCM_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    match manager.initialize() {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Get number of available ROCm devices
#[no_mangle]
pub extern "C" fn trustformers_rocm_get_device_count() -> c_int {
    let manager = match ROCM_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return -1,
    };

    if !manager.initialized {
        return -1;
    }

    manager.devices.len() as c_int
}

/// Get ROCm device information
#[no_mangle]
pub extern "C" fn trustformers_rocm_get_device_info(
    device_id: c_int,
    info_json: *mut *mut c_char,
) -> TrustformersError {
    if info_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let manager = match ROCM_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if !manager.initialized {
        return TrustformersError::RuntimeError;
    }

    let device_info = match manager.get_device_info(device_id) {
        Some(info) => info,
        None => return TrustformersError::InvalidParameter,
    };

    let info_json_data = serde_json::json!({
        "device_id": device_info.device_id,
        "name": device_info.name,
        "compute_capability": format!("{}.{}", device_info.compute_capability_major, device_info.compute_capability_minor),
        "total_memory_mb": device_info.total_memory_mb,
        "free_memory_mb": device_info.free_memory_mb,
        "multiprocessor_count": device_info.multiprocessor_count,
        "max_threads_per_block": device_info.max_threads_per_block,
        "max_shared_memory_per_block": device_info.max_shared_memory_per_block,
        "warp_size": device_info.warp_size,
        "max_grid_size": device_info.max_grid_size,
        "max_block_size": device_info.max_block_size,
        "pci_bus_id": device_info.pci_bus_id,
        "pci_device_id": device_info.pci_device_id,
        "clock_rate_khz": device_info.clock_rate_khz,
        "memory_clock_rate_khz": device_info.memory_clock_rate_khz,
        "memory_bus_width": device_info.memory_bus_width,
        "l2_cache_size": device_info.l2_cache_size,
    });

    let json_string = match serde_json::to_string_pretty(&info_json_data) {
        Ok(json) => json,
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        *info_json = string_to_c_str(json_string);
    }

    TrustformersError::Success
}

/// Set current ROCm device
#[no_mangle]
pub extern "C" fn trustformers_rocm_set_device(device_id: c_int) -> TrustformersError {
    let mut manager = match ROCM_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if !manager.initialized {
        return TrustformersError::RuntimeError;
    }

    match manager.set_current_device(device_id) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::InvalidParameter,
    }
}

/// Get current ROCm device
#[no_mangle]
pub extern "C" fn trustformers_rocm_get_current_device() -> c_int {
    let manager = match ROCM_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return -1,
    };

    manager.get_current_device().unwrap_or(-1)
}

/// Allocate ROCm tensor
#[no_mangle]
pub extern "C" fn trustformers_rocm_allocate_tensor(
    shape: *const usize,
    rank: c_int,
    dtype: c_int, // 0=f32, 1=f16, 2=i32, 3=i8, 4=u8, 5=f64, 6=i16, 7=i64
    device_id: c_int,
    tensor_handle: *mut usize,
) -> TrustformersError {
    if shape.is_null() || tensor_handle.is_null() || rank <= 0 {
        return TrustformersError::NullPointer;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank as usize) };

    let tensor_dtype = match dtype {
        0 => TensorDataType::Float32,
        1 => TensorDataType::Float16,
        2 => TensorDataType::Int32,
        3 => TensorDataType::Int8,
        4 => TensorDataType::UInt8,
        5 => TensorDataType::Float64,
        6 => TensorDataType::Int16,
        7 => TensorDataType::Int64,
        _ => return TrustformersError::InvalidParameter,
    };

    let tensor = match RocmOperations::allocate_tensor(shape_slice, tensor_dtype, device_id) {
        Ok(tensor) => tensor,
        Err(_) => return TrustformersError::OutOfMemory,
    };

    // Generate a unique handle and store tensor in the registry
    let handle = match ROCM_TENSOR_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersError::RuntimeError,
    };

    // Store tensor in the global registry
    match ROCM_TENSOR_REGISTRY.lock() {
        Ok(mut registry) => {
            registry.insert(handle, tensor);
        },
        Err(_) => return TrustformersError::RuntimeError,
    }

    unsafe {
        *tensor_handle = handle;
    }

    TrustformersError::Success
}

/// Free ROCm tensor
#[no_mangle]
pub extern "C" fn trustformers_rocm_free_tensor(tensor_handle: usize) -> TrustformersError {
    let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if let Some(tensor) = registry.remove(&tensor_handle) {
        // Free the tensor
        match RocmOperations::free_tensor(&tensor) {
            Ok(_) => TrustformersError::Success,
            Err(_) => TrustformersError::RuntimeError,
        }
    } else {
        TrustformersError::InvalidParameter
    }
}

/// Matrix multiplication on ROCm
#[no_mangle]
pub extern "C" fn trustformers_rocm_matrix_multiply(
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
    m: usize,
    n: usize,
    k: usize,
) -> TrustformersError {
    let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let a_tensor = match registry.get(&a_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    let b_tensor = match registry.get(&b_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    let mut c_tensor = match registry.get(&c_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    // Release the registry lock before calling matrix_multiply
    drop(registry);

    match RocmOperations::matrix_multiply(&a_tensor, &b_tensor, &mut c_tensor, m, n, k) {
        Ok(_) => {
            // Update the result tensor in the registry
            let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(c_handle, c_tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Copy data from host to ROCm device
#[no_mangle]
pub extern "C" fn trustformers_rocm_copy_host_to_device(
    host_data: *const c_float,
    tensor_handle: usize,
    size: usize,
) -> TrustformersError {
    if host_data.is_null() {
        return TrustformersError::NullPointer;
    }

    let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let mut tensor = match registry.get(&tensor_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    // Release the registry lock before the copy operation
    drop(registry);

    let host_slice = unsafe { std::slice::from_raw_parts(host_data, size) };

    match RocmOperations::copy_host_to_device(host_slice, &mut tensor) {
        Ok(_) => {
            // Update the tensor in the registry with modified tensor
            let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(tensor_handle, tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Copy data from ROCm device to host
#[no_mangle]
pub extern "C" fn trustformers_rocm_copy_device_to_host(
    tensor_handle: usize,
    host_data: *mut c_float,
    size: usize,
) -> TrustformersError {
    if host_data.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = match ROCM_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let tensor = match registry.get(&tensor_handle) {
        Some(tensor) => tensor,
        None => return TrustformersError::InvalidParameter,
    };

    let host_slice = unsafe { std::slice::from_raw_parts_mut(host_data, size) };

    match RocmOperations::copy_device_to_host(tensor, host_slice) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Check if ROCm is available
#[no_mangle]
pub extern "C" fn trustformers_rocm_is_available() -> c_int {
    // In real implementation, this would check for ROCm runtime and drivers
    // For simulation, check environment variable
    if std::env::var("HIP_VISIBLE_DEVICES").is_ok()
        || std::env::var("ROCR_VISIBLE_DEVICES").is_ok()
        || std::env::var("DISABLE_ROCM").is_err()
    {
        1
    } else {
        0
    }
}

/// Get ROCm memory usage
#[no_mangle]
pub extern "C" fn trustformers_rocm_get_memory_info(
    device_id: c_int,
    free_memory: *mut u64,
    total_memory: *mut u64,
) -> TrustformersError {
    if free_memory.is_null() || total_memory.is_null() {
        return TrustformersError::NullPointer;
    }

    let manager = match ROCM_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let device_info = match manager.get_device_info(device_id) {
        Some(info) => info,
        None => return TrustformersError::InvalidParameter,
    };

    unsafe {
        *free_memory = device_info.free_memory_mb * 1024 * 1024;
        *total_memory = device_info.total_memory_mb * 1024 * 1024;
    }

    TrustformersError::Success
}

/// Synchronize ROCm device (wait for all operations to complete)
#[no_mangle]
pub extern "C" fn trustformers_rocm_synchronize() -> TrustformersError {
    // In real implementation, this would call hipDeviceSynchronize()
    TrustformersError::Success
}

/// Apply activation function on ROCm device
#[no_mangle]
pub extern "C" fn trustformers_rocm_apply_activation(
    input_handle: usize,
    output_handle: usize,
    activation: *const c_char,
) -> TrustformersError {
    if activation.is_null() {
        return TrustformersError::NullPointer;
    }

    let activation_str = match c_str_to_string(activation) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let input_tensor = match registry.get(&input_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    let mut output_tensor = match registry.get(&output_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    // Release the registry lock before the operation
    drop(registry);

    match RocmOperations::apply_activation(&input_tensor, &mut output_tensor, &activation_str) {
        Ok(_) => {
            // Update the output tensor in the registry
            let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(output_handle, output_tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Tensor element-wise addition on ROCm device
#[no_mangle]
pub extern "C" fn trustformers_rocm_tensor_add(
    a_handle: usize,
    b_handle: usize,
    result_handle: usize,
) -> TrustformersError {
    let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let a_tensor = match registry.get(&a_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    let b_tensor = match registry.get(&b_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    let mut result_tensor = match registry.get(&result_handle) {
        Some(tensor) => tensor.clone(),
        None => return TrustformersError::InvalidParameter,
    };

    // Release the registry lock before the operation
    drop(registry);

    match RocmOperations::tensor_add(&a_tensor, &b_tensor, &mut result_tensor) {
        Ok(_) => {
            // Update the result tensor in the registry
            let mut registry = match ROCM_TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(result_handle, result_tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}
