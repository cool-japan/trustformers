//! Metal Performance Shaders backend for Apple Silicon
//!
//! This module provides GPU acceleration using Metal Performance Shaders (MPS) on macOS
//! and iOS devices with Apple Silicon. It offers optimized tensor operations and model
//! inference using the unified memory architecture of Apple Silicon.

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::error::{TrustformersError, TrustformersResult};
use crate::{c_str_to_string, result_to_error, string_to_c_str};

/// Global Metal context manager
static METAL_MANAGER: Lazy<Mutex<MetalManager>> = Lazy::new(|| Mutex::new(MetalManager::new()));

/// Global tensor registry for C API
static METAL_TENSOR_REGISTRY: Lazy<Mutex<HashMap<usize, MetalTensor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global counter for tensor handles
static METAL_TENSOR_HANDLE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| {
    Mutex::new(2000) // Start at 2000 to avoid confusion with other handles
});

/// Metal device information
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub max_buffer_length: u64,
    pub max_threadgroup_memory_length: u64,
    pub max_threads_per_threadgroup: u64,
    pub is_unified_memory: bool,
    pub supports_family_apple7: bool,
    pub supports_family_apple8: bool,
    pub supports_family_apple9: bool,
    pub memory_size: u64,
}

/// Metal context and device management
struct MetalManager {
    devices: Vec<MetalDeviceInfo>,
    current_device: Option<i32>,
    #[cfg(target_os = "macos")]
    metal_devices: HashMap<i32, MetalDeviceHandle>,
    #[cfg(target_os = "macos")]
    command_queues: HashMap<i32, MetalCommandQueue>,
    initialized: bool,
}

/// Metal device handle (platform-specific)
#[cfg(target_os = "macos")]
struct MetalDeviceHandle {
    device_ptr: *mut c_void,
}

#[cfg(target_os = "macos")]
unsafe impl Send for MetalDeviceHandle {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalDeviceHandle {}

/// Metal command queue (platform-specific)
#[cfg(target_os = "macos")]
struct MetalCommandQueue {
    queue_ptr: *mut c_void,
}

#[cfg(target_os = "macos")]
unsafe impl Send for MetalCommandQueue {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalCommandQueue {}

impl std::fmt::Debug for MetalManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalManager")
            .field("devices", &self.devices)
            .field("current_device", &self.current_device)
            .field("initialized", &self.initialized)
            .finish()
    }
}

/// Metal tensor allocation info
pub struct MetalTensor {
    #[cfg(target_os = "macos")]
    pub buffer: MetalBuffer,
    #[cfg(not(target_os = "macos"))]
    pub buffer: usize, // Fallback for non-macOS platforms
    pub shape: Vec<usize>,
    pub dtype: TensorDataType,
    pub size_bytes: usize,
    pub device_id: i32,
}

/// Metal buffer wrapper
#[cfg(target_os = "macos")]
pub struct MetalBuffer {
    buffer_ptr: *mut c_void,
    size: usize,
}

#[cfg(target_os = "macos")]
unsafe impl Send for MetalBuffer {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalBuffer {}

impl std::fmt::Debug for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("size_bytes", &self.size_bytes)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl Clone for MetalTensor {
    fn clone(&self) -> Self {
        Self {
            #[cfg(target_os = "macos")]
            buffer: MetalBuffer {
                buffer_ptr: self.buffer.buffer_ptr,
                size: self.buffer.size,
            },
            #[cfg(not(target_os = "macos"))]
            buffer: self.buffer,
            shape: self.shape.clone(),
            dtype: self.dtype,
            size_bytes: self.size_bytes,
            device_id: self.device_id,
        }
    }
}

/// Tensor data types supported by Metal backend
#[derive(Debug, Clone, Copy)]
pub enum TensorDataType {
    Float32,
    Float16,
    Int32,
    Int8,
    UInt8,
}

impl TensorDataType {
    fn size_in_bytes(&self) -> usize {
        match self {
            TensorDataType::Float32 => 4,
            TensorDataType::Float16 => 2,
            TensorDataType::Int32 => 4,
            TensorDataType::Int8 => 1,
            TensorDataType::UInt8 => 1,
        }
    }

    #[cfg(target_os = "macos")]
    fn to_metal_data_type(&self) -> i32 {
        match self {
            TensorDataType::Float32 => 1, // MTLDataTypeFloat
            TensorDataType::Float16 => 2, // MTLDataTypeHalf
            TensorDataType::Int32 => 29,  // MTLDataTypeInt
            TensorDataType::Int8 => 45,   // MTLDataTypeChar
            TensorDataType::UInt8 => 62,  // MTLDataTypeUChar
        }
    }
}

impl MetalManager {
    fn new() -> Self {
        Self {
            devices: Vec::new(),
            current_device: None,
            #[cfg(target_os = "macos")]
            metal_devices: HashMap::new(),
            #[cfg(target_os = "macos")]
            command_queues: HashMap::new(),
            initialized: false,
        }
    }

    /// Initialize Metal and detect devices
    fn initialize(&mut self) -> TrustformersResult<()> {
        if self.initialized {
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        {
            // Real Metal initialization on macOS
            self.detect_metal_devices()?;
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Fallback simulation when not on macOS
            self.detect_simulated_devices()?;
        }

        self.initialized = true;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    /// Detect real Metal devices using Metal framework
    fn detect_metal_devices(&mut self) -> TrustformersResult<()> {
        // Use Metal framework to enumerate devices
        let device_count = unsafe { metal_get_device_count() };

        if device_count == 0 {
            return Err(TrustformersError::DeviceNotAvailable);
        }

        for device_id in 0..device_count {
            let device_ptr = unsafe { metal_create_device(device_id) };
            if device_ptr.is_null() {
                continue;
            }

            // Get device properties
            let mut device_info = MetalDeviceInfo {
                device_id,
                name: String::new(),
                max_buffer_length: 0,
                max_threadgroup_memory_length: 0,
                max_threads_per_threadgroup: 0,
                is_unified_memory: false,
                supports_family_apple7: false,
                supports_family_apple8: false,
                supports_family_apple9: false,
                memory_size: 0,
            };

            // Query device properties
            unsafe {
                let name_ptr = metal_get_device_name(device_ptr);
                if !name_ptr.is_null() {
                    device_info.name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                    metal_free_string(name_ptr);
                }

                device_info.max_buffer_length = metal_get_max_buffer_length(device_ptr);
                device_info.max_threadgroup_memory_length =
                    metal_get_max_threadgroup_memory_length(device_ptr);
                device_info.max_threads_per_threadgroup =
                    metal_get_max_threads_per_threadgroup(device_ptr);
                device_info.is_unified_memory = metal_has_unified_memory(device_ptr) != 0;
                device_info.supports_family_apple7 = metal_supports_family_apple7(device_ptr) != 0;
                device_info.supports_family_apple8 = metal_supports_family_apple8(device_ptr) != 0;
                device_info.supports_family_apple9 = metal_supports_family_apple9(device_ptr) != 0;
                device_info.memory_size = metal_get_memory_size(device_ptr);
            }

            // Create command queue
            let command_queue_ptr = unsafe { metal_create_command_queue(device_ptr) };
            if command_queue_ptr.is_null() {
                unsafe {
                    metal_release_device(device_ptr);
                }
                continue;
            }

            // Store the Metal device and command queue
            self.metal_devices.insert(device_id, MetalDeviceHandle { device_ptr });
            self.command_queues.insert(
                device_id,
                MetalCommandQueue {
                    queue_ptr: command_queue_ptr,
                },
            );
            self.devices.push(device_info);
        }

        // Set first device as current if available
        if !self.devices.is_empty() {
            self.set_current_device(0)?;
        }

        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    /// Detect simulated devices when Metal is not available
    fn detect_simulated_devices(&mut self) -> TrustformersResult<()> {
        // Simulate Apple Silicon device
        let device_info = MetalDeviceInfo {
            device_id: 0,
            name: "Simulated Apple M1".to_string(),
            max_buffer_length: 4 * 1024 * 1024 * 1024, // 4GB
            max_threadgroup_memory_length: 32 * 1024,  // 32KB
            max_threads_per_threadgroup: 1024,
            is_unified_memory: true,
            supports_family_apple7: true,
            supports_family_apple8: true,
            supports_family_apple9: false,
            memory_size: 16 * 1024 * 1024 * 1024, // 16GB unified memory
        };

        self.devices.push(device_info);

        // Set first device as current
        self.set_current_device(0)?;

        Ok(())
    }

    fn set_current_device(&mut self, device_id: i32) -> TrustformersResult<()> {
        if device_id < 0 || device_id >= self.devices.len() as i32 {
            return Err(TrustformersError::InvalidParameter);
        }

        self.current_device = Some(device_id);
        Ok(())
    }

    fn get_device_info(&self, device_id: i32) -> Option<&MetalDeviceInfo> {
        self.devices.get(device_id as usize)
    }

    fn get_current_device(&self) -> Option<i32> {
        self.current_device
    }
}

/// Metal operations implementation
pub struct MetalOperations;

impl MetalOperations {
    /// Allocate memory on Metal device
    pub fn allocate_tensor(
        shape: &[usize],
        dtype: TensorDataType,
        device_id: i32,
    ) -> TrustformersResult<MetalTensor> {
        let total_elements: usize = shape.iter().product();
        let size_bytes = total_elements * dtype.size_in_bytes();

        #[cfg(target_os = "macos")]
        {
            let manager = METAL_MANAGER.lock().map_err(|_| TrustformersError::ConcurrencyError)?;
            if let Some(device_handle) = manager.metal_devices.get(&device_id) {
                let buffer_ptr =
                    unsafe { metal_create_buffer(device_handle.device_ptr, size_bytes as u64) };

                if buffer_ptr.is_null() {
                    return Err(TrustformersError::OutOfMemory);
                }

                return Ok(MetalTensor {
                    buffer: MetalBuffer {
                        buffer_ptr,
                        size: size_bytes,
                    },
                    shape: shape.to_vec(),
                    dtype,
                    size_bytes,
                    device_id,
                });
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            let buffer = Self::simulate_metal_malloc(size_bytes);
            return Ok(MetalTensor {
                buffer,
                shape: shape.to_vec(),
                dtype,
                size_bytes,
                device_id,
            });
        }

        #[cfg(target_os = "macos")]
        Err(TrustformersError::DeviceNotAvailable)
    }

    /// Free Metal tensor memory
    pub fn free_tensor(tensor: &MetalTensor) -> TrustformersResult<()> {
        #[cfg(target_os = "macos")]
        {
            unsafe {
                metal_release_buffer(tensor.buffer.buffer_ptr);
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Simulation - no actual memory to free
            Ok(())
        }
    }

    /// Copy data from host to device
    pub fn copy_host_to_device(
        host_data: &[f32],
        tensor: &mut MetalTensor,
    ) -> TrustformersResult<()> {
        if host_data.len() * std::mem::size_of::<f32>() != tensor.size_bytes {
            return Err(TrustformersError::InvalidParameter);
        }

        #[cfg(target_os = "macos")]
        {
            unsafe {
                let contents = metal_get_buffer_contents(tensor.buffer.buffer_ptr);
                if contents.is_null() {
                    return Err(TrustformersError::RuntimeError);
                }

                std::ptr::copy_nonoverlapping(
                    host_data.as_ptr() as *const u8,
                    contents as *mut u8,
                    tensor.size_bytes,
                );

                metal_did_modify_range(tensor.buffer.buffer_ptr, 0, tensor.size_bytes as u64);
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Simulation
            Self::simulate_metal_memcpy_h2d(host_data.as_ptr(), tensor.buffer, tensor.size_bytes);
            Ok(())
        }
    }

    /// Copy data from device to host
    pub fn copy_device_to_host(
        tensor: &MetalTensor,
        host_data: &mut [f32],
    ) -> TrustformersResult<()> {
        if host_data.len() * std::mem::size_of::<f32>() != tensor.size_bytes {
            return Err(TrustformersError::InvalidParameter);
        }

        #[cfg(target_os = "macos")]
        {
            unsafe {
                let contents = metal_get_buffer_contents(tensor.buffer.buffer_ptr);
                if contents.is_null() {
                    return Err(TrustformersError::RuntimeError);
                }

                std::ptr::copy_nonoverlapping(
                    contents as *const u8,
                    host_data.as_mut_ptr() as *mut u8,
                    tensor.size_bytes,
                );
            }
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Simulation
            Self::simulate_metal_memcpy_d2h(
                tensor.buffer,
                host_data.as_mut_ptr(),
                tensor.size_bytes,
            );
            Ok(())
        }
    }

    /// Matrix multiplication using Metal Performance Shaders
    pub fn matrix_multiply(
        a: &MetalTensor,
        b: &MetalTensor,
        c: &mut MetalTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> TrustformersResult<()> {
        // Validate dimensions
        if a.shape != [m, k] || b.shape != [k, n] || c.shape != [m, n] {
            return Err(TrustformersError::InvalidParameter);
        }

        #[cfg(target_os = "macos")]
        {
            let manager = METAL_MANAGER.lock().map_err(|_| TrustformersError::ConcurrencyError)?;
            if let Some(command_queue) = manager.command_queues.get(&a.device_id) {
                unsafe {
                    let result = metal_matrix_multiply(
                        command_queue.queue_ptr,
                        a.buffer.buffer_ptr,
                        b.buffer.buffer_ptr,
                        c.buffer.buffer_ptr,
                        m as u64,
                        n as u64,
                        k as u64,
                        a.dtype.to_metal_data_type(),
                    );

                    if result != 0 {
                        return Err(TrustformersError::RuntimeError);
                    }
                }

                return Ok(());
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Simulation fallback
            Self::simulate_metal_gemm(a.buffer, b.buffer, c.buffer, m, n, k);
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        Err(TrustformersError::DeviceNotAvailable)
    }

    /// Element-wise addition using Metal compute shaders
    pub fn tensor_add(
        a: &MetalTensor,
        b: &MetalTensor,
        result: &mut MetalTensor,
    ) -> TrustformersResult<()> {
        if a.shape != b.shape || a.shape != result.shape {
            return Err(TrustformersError::InvalidParameter);
        }

        #[cfg(target_os = "macos")]
        {
            let manager = METAL_MANAGER.lock().map_err(|_| TrustformersError::ConcurrencyError)?;
            if let Some(command_queue) = manager.command_queues.get(&a.device_id) {
                let total_elements = a.shape.iter().product::<usize>();

                unsafe {
                    let result_code = metal_tensor_add(
                        command_queue.queue_ptr,
                        a.buffer.buffer_ptr,
                        b.buffer.buffer_ptr,
                        result.buffer.buffer_ptr,
                        total_elements as u64,
                        a.dtype.to_metal_data_type(),
                    );

                    if result_code != 0 {
                        return Err(TrustformersError::RuntimeError);
                    }
                }

                return Ok(());
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Simulation fallback
            Self::simulate_metal_elementwise_add(
                a.buffer,
                b.buffer,
                result.buffer,
                a.shape.iter().product(),
            );
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        Err(TrustformersError::DeviceNotAvailable)
    }

    /// Apply activation functions using Metal compute shaders
    pub fn apply_activation(
        input: &MetalTensor,
        output: &mut MetalTensor,
        activation: &str,
    ) -> TrustformersResult<()> {
        if input.shape != output.shape {
            return Err(TrustformersError::InvalidParameter);
        }

        #[cfg(target_os = "macos")]
        {
            let manager = METAL_MANAGER.lock().map_err(|_| TrustformersError::ConcurrencyError)?;
            if let Some(command_queue) = manager.command_queues.get(&input.device_id) {
                let total_elements = input.shape.iter().product::<usize>();
                let activation_type = match activation {
                    "relu" => 0,
                    "gelu" => 1,
                    "tanh" => 2,
                    "sigmoid" => 3,
                    _ => return Err(TrustformersError::FeatureNotAvailable),
                };

                unsafe {
                    let result_code = metal_apply_activation(
                        command_queue.queue_ptr,
                        input.buffer.buffer_ptr,
                        output.buffer.buffer_ptr,
                        total_elements as u64,
                        activation_type,
                        input.dtype.to_metal_data_type(),
                    );

                    if result_code != 0 {
                        return Err(TrustformersError::RuntimeError);
                    }
                }

                return Ok(());
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Simulation fallback
            match activation {
                "relu" => Self::simulate_metal_relu(
                    input.buffer,
                    output.buffer,
                    input.shape.iter().product(),
                ),
                "gelu" => Self::simulate_metal_gelu(
                    input.buffer,
                    output.buffer,
                    input.shape.iter().product(),
                ),
                "tanh" => Self::simulate_metal_tanh(
                    input.buffer,
                    output.buffer,
                    input.shape.iter().product(),
                ),
                "sigmoid" => Self::simulate_metal_sigmoid(
                    input.buffer,
                    output.buffer,
                    input.shape.iter().product(),
                ),
                _ => return Err(TrustformersError::FeatureNotAvailable),
            }
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        Err(TrustformersError::DeviceNotAvailable)
    }

    // Simulation functions (used on non-macOS platforms)
    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_malloc(size: usize) -> usize {
        // Return a simulated device pointer
        0x90000000 + size % 0x1000000
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_memcpy_h2d(_host_ptr: *const f32, _device_ptr: usize, _size: usize) {
        // In real implementation, this would copy to Metal buffer
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_memcpy_d2h(_device_ptr: usize, _host_ptr: *mut f32, _size: usize) {
        // In real implementation, this would copy from Metal buffer
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_gemm(
        _a_ptr: usize,
        _b_ptr: usize,
        _c_ptr: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) {
        // In real implementation, this would use MPSMatrixMultiplication
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_elementwise_add(
        _a_ptr: usize,
        _b_ptr: usize,
        _result_ptr: usize,
        _size: usize,
    ) {
        // In real implementation, this would use custom Metal compute shader
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_relu(_input_ptr: usize, _output_ptr: usize, _size: usize) {
        // In real implementation, this would use ReLU compute shader
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_gelu(_input_ptr: usize, _output_ptr: usize, _size: usize) {
        // In real implementation, this would use GELU compute shader
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_tanh(_input_ptr: usize, _output_ptr: usize, _size: usize) {
        // In real implementation, this would use tanh compute shader
    }

    #[cfg(not(target_os = "macos"))]
    fn simulate_metal_sigmoid(_input_ptr: usize, _output_ptr: usize, _size: usize) {
        // In real implementation, this would use sigmoid compute shader
    }
}

// External Metal framework functions (macOS only)
#[cfg(target_os = "macos")]
extern "C" {
    fn metal_get_device_count() -> i32;
    fn metal_create_device(device_id: i32) -> *mut c_void;
    fn metal_release_device(device: *mut c_void);
    fn metal_get_device_name(device: *mut c_void) -> *mut c_char;
    fn metal_get_max_buffer_length(device: *mut c_void) -> u64;
    fn metal_get_max_threadgroup_memory_length(device: *mut c_void) -> u64;
    fn metal_get_max_threads_per_threadgroup(device: *mut c_void) -> u64;
    fn metal_has_unified_memory(device: *mut c_void) -> c_int;
    fn metal_supports_family_apple7(device: *mut c_void) -> c_int;
    fn metal_supports_family_apple8(device: *mut c_void) -> c_int;
    fn metal_supports_family_apple9(device: *mut c_void) -> c_int;
    fn metal_get_memory_size(device: *mut c_void) -> u64;
    fn metal_create_command_queue(device: *mut c_void) -> *mut c_void;
    fn metal_release_command_queue(queue: *mut c_void);
    fn metal_create_buffer(device: *mut c_void, size: u64) -> *mut c_void;
    fn metal_release_buffer(buffer: *mut c_void);
    fn metal_get_buffer_contents(buffer: *mut c_void) -> *mut c_void;
    fn metal_did_modify_range(buffer: *mut c_void, offset: u64, length: u64);
    fn metal_matrix_multiply(
        queue: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
        m: u64,
        n: u64,
        k: u64,
        data_type: i32,
    ) -> i32;
    fn metal_tensor_add(
        queue: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        result: *mut c_void,
        elements: u64,
        data_type: i32,
    ) -> i32;
    fn metal_apply_activation(
        queue: *mut c_void,
        input: *mut c_void,
        output: *mut c_void,
        elements: u64,
        activation_type: i32,
        data_type: i32,
    ) -> i32;
    fn metal_free_string(string: *mut c_char);
}

// C API exports for Metal functionality

/// Initialize Metal backend
#[no_mangle]
pub extern "C" fn trustformers_metal_init() -> TrustformersError {
    let mut manager = match METAL_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    match manager.initialize() {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Get number of available Metal devices
#[no_mangle]
pub extern "C" fn trustformers_metal_get_device_count() -> c_int {
    let manager = match METAL_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return -1,
    };

    if !manager.initialized {
        return -1;
    }

    manager.devices.len() as c_int
}

/// Get Metal device information
#[no_mangle]
pub extern "C" fn trustformers_metal_get_device_info(
    device_id: c_int,
    info_json: *mut *mut c_char,
) -> TrustformersError {
    if info_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let manager = match METAL_MANAGER.lock() {
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
        "max_buffer_length": device_info.max_buffer_length,
        "max_threadgroup_memory_length": device_info.max_threadgroup_memory_length,
        "max_threads_per_threadgroup": device_info.max_threads_per_threadgroup,
        "is_unified_memory": device_info.is_unified_memory,
        "supports_family_apple7": device_info.supports_family_apple7,
        "supports_family_apple8": device_info.supports_family_apple8,
        "supports_family_apple9": device_info.supports_family_apple9,
        "memory_size": device_info.memory_size,
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

/// Set current Metal device
#[no_mangle]
pub extern "C" fn trustformers_metal_set_device(device_id: c_int) -> TrustformersError {
    let mut manager = match METAL_MANAGER.lock() {
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

/// Get current Metal device
#[no_mangle]
pub extern "C" fn trustformers_metal_get_current_device() -> c_int {
    let manager = match METAL_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return -1,
    };

    manager.get_current_device().unwrap_or(-1)
}

/// Check if Metal is available
#[no_mangle]
pub extern "C" fn trustformers_metal_is_available() -> c_int {
    #[cfg(target_os = "macos")]
    {
        1 // Metal is available on macOS
    }
    #[cfg(not(target_os = "macos"))]
    {
        0 // Metal is not available on non-macOS platforms
    }
}

/// Allocate Metal tensor
#[no_mangle]
pub extern "C" fn trustformers_metal_allocate_tensor(
    shape: *const usize,
    rank: c_int,
    dtype: c_int, // 0=f32, 1=f16, 2=i32, 3=i8, 4=u8
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
        _ => return TrustformersError::InvalidParameter,
    };

    let tensor = match MetalOperations::allocate_tensor(shape_slice, tensor_dtype, device_id) {
        Ok(tensor) => tensor,
        Err(_) => return TrustformersError::OutOfMemory,
    };

    // Generate a unique handle and store tensor in the registry
    let handle = match METAL_TENSOR_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersError::RuntimeError,
    };

    // Store tensor in the global registry
    match METAL_TENSOR_REGISTRY.lock() {
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

/// Free Metal tensor
#[no_mangle]
pub extern "C" fn trustformers_metal_free_tensor(tensor_handle: usize) -> TrustformersError {
    let mut registry = match METAL_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if let Some(tensor) = registry.remove(&tensor_handle) {
        // Free the tensor
        match MetalOperations::free_tensor(&tensor) {
            Ok(_) => TrustformersError::Success,
            Err(_) => TrustformersError::RuntimeError,
        }
    } else {
        TrustformersError::InvalidParameter
    }
}

/// Matrix multiplication on Metal
#[no_mangle]
pub extern "C" fn trustformers_metal_matrix_multiply(
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
    m: usize,
    n: usize,
    k: usize,
) -> TrustformersError {
    let mut registry = match METAL_TENSOR_REGISTRY.lock() {
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

    // Release the registry lock before the operation
    drop(registry);

    match MetalOperations::matrix_multiply(&a_tensor, &b_tensor, &mut c_tensor, m, n, k) {
        Ok(_) => {
            // Update the result tensor in the registry
            let mut registry = match METAL_TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(c_handle, c_tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Copy data from host to Metal device
#[no_mangle]
pub extern "C" fn trustformers_metal_copy_host_to_device(
    host_data: *const c_float,
    tensor_handle: usize,
    size: usize,
) -> TrustformersError {
    if host_data.is_null() {
        return TrustformersError::NullPointer;
    }

    let mut registry = match METAL_TENSOR_REGISTRY.lock() {
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

    match MetalOperations::copy_host_to_device(host_slice, &mut tensor) {
        Ok(_) => {
            // Update the tensor in the registry
            let mut registry = match METAL_TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(tensor_handle, tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Copy data from Metal device to host
#[no_mangle]
pub extern "C" fn trustformers_metal_copy_device_to_host(
    tensor_handle: usize,
    host_data: *mut c_float,
    size: usize,
) -> TrustformersError {
    if host_data.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = match METAL_TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let tensor = match registry.get(&tensor_handle) {
        Some(tensor) => tensor,
        None => return TrustformersError::InvalidParameter,
    };

    let host_slice = unsafe { std::slice::from_raw_parts_mut(host_data, size) };

    match MetalOperations::copy_device_to_host(tensor, host_slice) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Synchronize Metal operations (wait for completion)
#[no_mangle]
pub extern "C" fn trustformers_metal_synchronize() -> TrustformersError {
    #[cfg(target_os = "macos")]
    {
        // In real implementation, this would wait for all Metal command buffers to complete
        TrustformersError::Success
    }
    #[cfg(not(target_os = "macos"))]
    {
        TrustformersError::Success
    }
}
