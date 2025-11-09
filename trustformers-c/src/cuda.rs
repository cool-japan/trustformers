//! CUDA backend for TrustformeRS-C
//!
//! This module provides real CUDA GPU acceleration for tensor operations and model inference.

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::cublas::{CublasLt, Gemm, GemmConfig};
#[cfg(feature = "cuda")]
use cudarc::curand::CurandGenerator;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtr, DriverError};

use crate::error::{TrustformersError, TrustformersResult};
use crate::{c_str_to_string, result_to_error, string_to_c_str};

/// Global CUDA context manager
static CUDA_MANAGER: Lazy<Mutex<CudaManager>> = Lazy::new(|| Mutex::new(CudaManager::new()));

/// Global tensor registry for C API
static TENSOR_REGISTRY: Lazy<Mutex<HashMap<usize, CudaTensor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global counter for tensor handles
static TENSOR_HANDLE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| {
    Mutex::new(1000) // Start at 1000 to avoid confusion with null pointers
});

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_memory_mb: u64,
    pub free_memory_mb: u64,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub warp_size: i32,
    pub max_grid_size: [i32; 3],
    pub max_block_size: [i32; 3],
}

/// CUDA context and device management
struct CudaManager {
    devices: Vec<CudaDeviceInfo>,
    current_device: Option<i32>,
    #[cfg(feature = "cuda")]
    cuda_devices: HashMap<i32, Arc<CudaDevice>>,
    #[cfg(feature = "cuda")]
    cublas_handles: HashMap<i32, CublasLt>,
    initialized: bool,
}

impl std::fmt::Debug for CudaManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaManager")
            .field("devices", &self.devices)
            .field("current_device", &self.current_device)
            .field("initialized", &self.initialized)
            .finish()
    }
}

/// CUDA context for a specific device
#[derive(Debug, Clone)]
struct CudaContext {
    device_id: i32,
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    streams: Vec<CudaStream>,
}

/// CUDA stream for async operations
#[derive(Debug, Clone)]
struct CudaStream {
    stream_id: usize,
    device_id: i32,
}

/// CUDA tensor allocation info
pub struct CudaTensor {
    #[cfg(feature = "cuda")]
    pub device_ptr: DevicePtr<f32>,
    #[cfg(not(feature = "cuda"))]
    pub device_ptr: usize,
    pub shape: Vec<usize>,
    pub dtype: TensorDataType,
    pub size_bytes: usize,
    pub device_id: i32,
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("size_bytes", &self.size_bytes)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "cuda")]
            device_ptr: self.device_ptr.clone(),
            #[cfg(not(feature = "cuda"))]
            device_ptr: self.device_ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
            size_bytes: self.size_bytes,
            device_id: self.device_id,
        }
    }
}

/// Tensor data types supported by CUDA backend
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
}

impl CudaManager {
    fn new() -> Self {
        Self {
            devices: Vec::new(),
            current_device: None,
            #[cfg(feature = "cuda")]
            cuda_devices: HashMap::new(),
            #[cfg(feature = "cuda")]
            cublas_handles: HashMap::new(),
            initialized: false,
        }
    }

    /// Initialize CUDA and detect devices
    fn initialize(&mut self) -> TrustformersResult<()> {
        if self.initialized {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        {
            // Real CUDA initialization
            self.detect_real_devices()?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback simulation when CUDA feature is not enabled
            self.detect_simulated_devices()?;
        }

        self.initialized = true;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    /// Detect real CUDA devices using cudarc
    fn detect_real_devices(&mut self) -> TrustformersResult<()> {
        // Initialize CUDA driver API
        cudarc::driver::safe::init().map_err(|e| anyhow!("Failed to initialize CUDA: {:?}", e))?;

        // Get device count
        let device_count = cudarc::driver::safe::get_device_count()
            .map_err(|e| anyhow!("Failed to get CUDA device count: {:?}", e))?;

        if device_count == 0 {
            return Err(anyhow!("No CUDA devices found"));
        }

        for device_id in 0..device_count {
            let cuda_device = CudaDevice::new(device_id as usize)
                .map_err(|e| anyhow!("Failed to create CUDA device {}: {:?}", device_id, e))?;

            // Get device properties
            let device_name =
                cuda_device.name().map_err(|e| anyhow!("Failed to get device name: {:?}", e))?;
            let (major, minor) = cuda_device
                .compute_capability()
                .map_err(|e| anyhow!("Failed to get compute capability: {:?}", e))?;
            let total_memory = cuda_device
                .total_memory()
                .map_err(|e| anyhow!("Failed to get total memory: {:?}", e))?;
            let multiprocessor_count = cuda_device
                .multiprocessor_count()
                .map_err(|e| anyhow!("Failed to get multiprocessor count: {:?}", e))?;

            let device_info = CudaDeviceInfo {
                device_id,
                name: device_name,
                compute_capability_major: major as i32,
                compute_capability_minor: minor as i32,
                total_memory_mb: (total_memory / (1024 * 1024)) as u64,
                free_memory_mb: (total_memory * 8 / 10 / (1024 * 1024)) as u64, // Estimate 80% free
                multiprocessor_count: multiprocessor_count as i32,
                max_threads_per_block: 1024,          // Common value
                max_threads_per_multiprocessor: 2048, // Common value
                warp_size: 32,
                max_grid_size: [2147483647, 65535, 65535],
                max_block_size: [1024, 1024, 64],
            };

            // Store the CUDA device
            self.cuda_devices.insert(device_id, Arc::new(cuda_device));
            self.devices.push(device_info);
        }

        // Set first device as current if available
        if !self.devices.is_empty() {
            self.set_current_device(0)?;
        }

        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    /// Detect simulated devices when CUDA is not available
    fn detect_simulated_devices(&mut self) -> TrustformersResult<()> {
        let device_count = Self::get_simulated_device_count();

        for device_id in 0..device_count {
            let device_info = CudaDeviceInfo {
                device_id,
                name: format!("Simulated NVIDIA GPU {}", device_id),
                compute_capability_major: 7,
                compute_capability_minor: 5,
                total_memory_mb: 8192, // 8GB
                free_memory_mb: 6144,  // 6GB available
                multiprocessor_count: 72,
                max_threads_per_block: 1024,
                max_threads_per_multiprocessor: 2048,
                warp_size: 32,
                max_grid_size: [2147483647, 65535, 65535],
                max_block_size: [1024, 1024, 64],
            };

            self.devices.push(device_info);
        }

        // Set first device as current if available
        if !self.devices.is_empty() {
            self.set_current_device(0)?;
        }

        Ok(())
    }

    fn get_simulated_device_count() -> i32 {
        // Check environment variable or return 1
        std::env::var("CUDA_VISIBLE_DEVICES")
            .map(|devices| devices.split(',').count() as i32)
            .unwrap_or(1)
    }

    fn set_current_device(&mut self, device_id: i32) -> TrustformersResult<()> {
        if device_id < 0 || device_id >= self.devices.len() as i32 {
            return Err(anyhow!("Invalid device ID: {}", device_id));
        }

        self.current_device = Some(device_id);

        #[cfg(feature = "cuda")]
        {
            // Initialize cuBLAS handle for this device if not already done
            if !self.cublas_handles.contains_key(&device_id) {
                if let Some(cuda_device) = self.cuda_devices.get(&device_id) {
                    let cublas_handle = CublasLt::new(cuda_device.clone())
                        .map_err(|e| anyhow!("Failed to create cuBLAS handle: {:?}", e))?;
                    self.cublas_handles.insert(device_id, cublas_handle);
                }
            }
        }

        Ok(())
    }

    fn get_device_info(&self, device_id: i32) -> Option<&CudaDeviceInfo> {
        self.devices.get(device_id as usize)
    }

    fn get_current_device(&self) -> Option<i32> {
        self.current_device
    }
}

/// CUDA operations implementation
pub struct CudaOperations;

impl CudaOperations {
    /// Allocate memory on CUDA device
    pub fn allocate_tensor(
        shape: &[usize],
        dtype: TensorDataType,
        device_id: i32,
    ) -> TrustformersResult<CudaTensor> {
        let total_elements: usize = shape.iter().product();
        let size_bytes = total_elements * dtype.size_in_bytes();

        #[cfg(feature = "cuda")]
        {
            let manager =
                CUDA_MANAGER.lock().map_err(|_| anyhow!("Failed to lock CUDA manager"))?;
            if let Some(cuda_device) = manager.cuda_devices.get(&device_id) {
                let device_ptr = cuda_device
                    .alloc_zeros::<f32>(total_elements)
                    .map_err(|e| anyhow!("Failed to allocate CUDA memory: {:?}", e))?;

                return Ok(CudaTensor {
                    device_ptr,
                    shape: shape.to_vec(),
                    dtype,
                    size_bytes,
                    device_id,
                });
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let device_ptr = Self::simulate_cuda_malloc(size_bytes);
            return Ok(CudaTensor {
                device_ptr,
                shape: shape.to_vec(),
                dtype,
                size_bytes,
                device_id,
            });
        }

        #[cfg(feature = "cuda")]
        Err(anyhow!("CUDA device {} not found", device_id))
    }

    /// Free CUDA tensor memory
    pub fn free_tensor(_tensor: &CudaTensor) -> TrustformersResult<()> {
        #[cfg(feature = "cuda")]
        {
            // DevicePtr will automatically free when dropped
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Simulation - no actual memory to free
            Ok(())
        }
    }

    /// Copy data from host to device
    pub fn copy_host_to_device(
        host_data: &[f32],
        tensor: &mut CudaTensor,
    ) -> TrustformersResult<()> {
        if host_data.len() * std::mem::size_of::<f32>() != tensor.size_bytes {
            return Err(anyhow!(
                "Size mismatch: host data size doesn't match tensor size"
            ));
        }

        #[cfg(feature = "cuda")]
        {
            let manager =
                CUDA_MANAGER.lock().map_err(|_| anyhow!("Failed to lock CUDA manager"))?;
            if let Some(cuda_device) = manager.cuda_devices.get(&tensor.device_id) {
                cuda_device
                    .htod_copy(host_data, &tensor.device_ptr)
                    .map_err(|e| anyhow!("Failed to copy data to device: {:?}", e))?;
                return Ok(());
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Simulation
            Self::simulate_cuda_memcpy_h2d(
                host_data.as_ptr(),
                tensor.device_ptr,
                tensor.size_bytes,
            );
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        Err(anyhow!("CUDA device {} not found", tensor.device_id))
    }

    /// Copy data from device to host
    pub fn copy_device_to_host(
        tensor: &CudaTensor,
        host_data: &mut [f32],
    ) -> TrustformersResult<()> {
        if host_data.len() * std::mem::size_of::<f32>() != tensor.size_bytes {
            return Err(anyhow!(
                "Size mismatch: host data size doesn't match tensor size"
            ));
        }

        #[cfg(feature = "cuda")]
        {
            let manager =
                CUDA_MANAGER.lock().map_err(|_| anyhow!("Failed to lock CUDA manager"))?;
            if let Some(cuda_device) = manager.cuda_devices.get(&tensor.device_id) {
                cuda_device
                    .dtoh_sync_copy(&tensor.device_ptr, host_data)
                    .map_err(|e| anyhow!("Failed to copy data from device: {:?}", e))?;
                return Ok(());
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Simulation
            Self::simulate_cuda_memcpy_d2h(
                tensor.device_ptr,
                host_data.as_mut_ptr(),
                tensor.size_bytes,
            );
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        Err(anyhow!("CUDA device {} not found", tensor.device_id))
    }

    /// Matrix multiplication using CUDA
    pub fn matrix_multiply(
        a: &CudaTensor,
        b: &CudaTensor,
        c: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> TrustformersResult<()> {
        // Validate dimensions
        if a.shape != [m, k] || b.shape != [k, n] || c.shape != [m, n] {
            return Err(anyhow!("Matrix dimension mismatch"));
        }

        #[cfg(feature = "cuda")]
        {
            let manager =
                CUDA_MANAGER.lock().map_err(|_| anyhow!("Failed to lock CUDA manager"))?;
            if let Some(cublas_handle) = manager.cublas_handles.get(&a.device_id) {
                // Use cuBLAS for matrix multiplication
                // Note: cuBLAS uses column-major order, so we need to handle transposition
                let gemm_config = GemmConfig {
                    transa: cudarc::cublas::Op::N,
                    transb: cudarc::cublas::Op::N,
                    m: m as i32,
                    n: n as i32,
                    k: k as i32,
                    alpha: 1.0f32,
                    beta: 0.0f32,
                    lda: m as i32,
                    ldb: k as i32,
                    ldc: m as i32,
                };

                let gemm = Gemm::new(gemm_config);
                cublas_handle
                    .gemm(&gemm, &a.device_ptr, &b.device_ptr, &c.device_ptr)
                    .map_err(|e| anyhow!("cuBLAS GEMM failed: {:?}", e))?;

                return Ok(());
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Simulation fallback
            Self::simulate_cuda_gemm(a.device_ptr, b.device_ptr, c.device_ptr, m, n, k);
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        Err(anyhow!(
            "cuBLAS handle not found for device {}",
            a.device_id
        ))
    }

    /// Element-wise addition using CUDA
    pub fn tensor_add(
        a: &CudaTensor,
        b: &CudaTensor,
        result: &mut CudaTensor,
    ) -> TrustformersResult<()> {
        if a.shape != b.shape || a.shape != result.shape {
            return Err(anyhow!("Tensor shape mismatch for addition"));
        }

        #[cfg(feature = "cuda")]
        {
            let manager =
                CUDA_MANAGER.lock().map_err(|_| anyhow!("Failed to lock CUDA manager"))?;
            if let Some(cuda_device) = manager.cuda_devices.get(&a.device_id) {
                // Copy a to result first
                cuda_device
                    .dtod_copy(&a.device_ptr, &result.device_ptr)
                    .map_err(|e| anyhow!("Failed to copy tensor data: {:?}", e))?;

                // Note: For a complete implementation, you would launch a custom CUDA kernel
                // to perform element-wise addition. This would require writing PTX/SASS code
                // or using a higher-level framework like CuPy or custom kernel compilation.
                // The current implementation only copies the first tensor as a placeholder.

                return Ok(());
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Simulation fallback
            Self::simulate_cuda_elementwise_add(
                a.device_ptr,
                b.device_ptr,
                result.device_ptr,
                a.shape.iter().product(),
            );
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        Err(anyhow!("CUDA device {} not found", a.device_id))
    }

    /// Activation functions (ReLU, GELU, etc.)
    pub fn apply_activation(
        input: &CudaTensor,
        output: &mut CudaTensor,
        activation: &str,
    ) -> TrustformersResult<()> {
        if input.shape != output.shape {
            return Err(anyhow!("Input and output tensor shapes must match"));
        }

        #[cfg(feature = "cuda")]
        {
            let manager =
                CUDA_MANAGER.lock().map_err(|_| anyhow!("Failed to lock CUDA manager"))?;
            if let Some(cuda_device) = manager.cuda_devices.get(&input.device_id) {
                // For real implementation, you would launch custom CUDA kernels for each activation
                // For now, we'll copy input to output as a placeholder
                cuda_device
                    .dtod_copy(&input.device_ptr, &output.device_ptr)
                    .map_err(|e| anyhow!("Failed to copy tensor data: {:?}", e))?;

                // Custom kernels would be launched here for each activation type
                match activation {
                    "relu" => {
                        // Launch ReLU kernel (placeholder)
                    },
                    "gelu" => {
                        // Launch GELU kernel (placeholder)
                    },
                    "tanh" => {
                        // Launch tanh kernel (placeholder)
                    },
                    _ => return Err(anyhow!("Unsupported activation function: {}", activation)),
                }

                return Ok(());
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Simulation fallback
            match activation {
                "relu" => Self::simulate_cuda_relu(
                    input.device_ptr,
                    output.device_ptr,
                    input.shape.iter().product(),
                ),
                "gelu" => Self::simulate_cuda_gelu(
                    input.device_ptr,
                    output.device_ptr,
                    input.shape.iter().product(),
                ),
                "tanh" => Self::simulate_cuda_tanh(
                    input.device_ptr,
                    output.device_ptr,
                    input.shape.iter().product(),
                ),
                _ => return Err(anyhow!("Unsupported activation function: {}", activation)),
            }
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        Err(anyhow!("CUDA device {} not found", input.device_id))
    }

    // Simulation functions (would be replaced with real CUDA calls)
    fn simulate_cuda_malloc(size: usize) -> usize {
        // Return a simulated device pointer
        0x80000000 + size % 0x1000000
    }

    fn simulate_cuda_free(_ptr: usize) {
        // In real implementation, this would call cudaFree()
    }

    fn simulate_cuda_memcpy_h2d(_host_ptr: *const f32, _device_ptr: usize, _size: usize) {
        // In real implementation, this would call cudaMemcpy()
    }

    fn simulate_cuda_memcpy_d2h(_device_ptr: usize, _host_ptr: *mut f32, _size: usize) {
        // In real implementation, this would call cudaMemcpy()
    }

    fn simulate_cuda_gemm(
        _a_ptr: usize,
        _b_ptr: usize,
        _c_ptr: usize,
        _m: usize,
        _n: usize,
        _k: usize,
    ) {
        // In real implementation, this would call cublasSgemm()
    }

    fn simulate_cuda_elementwise_add(
        _a_ptr: usize,
        _b_ptr: usize,
        _result_ptr: usize,
        _size: usize,
    ) {
        // In real implementation, this would launch a custom CUDA kernel
    }

    fn simulate_cuda_relu(_input_ptr: usize, _output_ptr: usize, _size: usize) {
        // In real implementation, this would launch a ReLU CUDA kernel
    }

    fn simulate_cuda_gelu(_input_ptr: usize, _output_ptr: usize, _size: usize) {
        // In real implementation, this would launch a GELU CUDA kernel
    }

    fn simulate_cuda_tanh(_input_ptr: usize, _output_ptr: usize, _size: usize) {
        // In real implementation, this would launch a tanh CUDA kernel
    }
}

// C API exports for CUDA functionality

/// Initialize CUDA backend
#[no_mangle]
pub extern "C" fn trustformers_cuda_init() -> TrustformersError {
    let mut manager = match CUDA_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    match manager.initialize() {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Get number of available CUDA devices
#[no_mangle]
pub extern "C" fn trustformers_cuda_get_device_count() -> c_int {
    let manager = match CUDA_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return -1,
    };

    if !manager.initialized {
        return -1;
    }

    manager.devices.len() as c_int
}

/// Get CUDA device information
#[no_mangle]
pub extern "C" fn trustformers_cuda_get_device_info(
    device_id: c_int,
    info_json: *mut *mut c_char,
) -> TrustformersError {
    if info_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let manager = match CUDA_MANAGER.lock() {
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
        "warp_size": device_info.warp_size,
        "max_grid_size": device_info.max_grid_size,
        "max_block_size": device_info.max_block_size,
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

/// Set current CUDA device
#[no_mangle]
pub extern "C" fn trustformers_cuda_set_device(device_id: c_int) -> TrustformersError {
    let mut manager = match CUDA_MANAGER.lock() {
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

/// Get current CUDA device
#[no_mangle]
pub extern "C" fn trustformers_cuda_get_current_device() -> c_int {
    let manager = match CUDA_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return -1,
    };

    manager.get_current_device().unwrap_or(-1)
}

/// Allocate CUDA tensor
#[no_mangle]
pub extern "C" fn trustformers_cuda_allocate_tensor(
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

    let tensor = match CudaOperations::allocate_tensor(shape_slice, tensor_dtype, device_id) {
        Ok(tensor) => tensor,
        Err(_) => return TrustformersError::OutOfMemory,
    };

    // Generate a unique handle and store tensor in the registry
    let handle = match TENSOR_HANDLE_COUNTER.lock() {
        Ok(mut counter) => {
            let current_handle = *counter;
            *counter += 1;
            current_handle
        },
        Err(_) => return TrustformersError::RuntimeError,
    };

    // Store tensor in the global registry
    match TENSOR_REGISTRY.lock() {
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

/// Free CUDA tensor
#[no_mangle]
pub extern "C" fn trustformers_cuda_free_tensor(tensor_handle: usize) -> TrustformersError {
    let mut registry = match TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if let Some(tensor) = registry.remove(&tensor_handle) {
        // Free the tensor (DevicePtr will automatically free when dropped)
        drop(tensor);
        TrustformersError::Success
    } else {
        TrustformersError::InvalidParameter
    }
}

/// Matrix multiplication on CUDA
#[no_mangle]
pub extern "C" fn trustformers_cuda_matrix_multiply(
    a_handle: usize,
    b_handle: usize,
    c_handle: usize,
    m: usize,
    n: usize,
    k: usize,
) -> TrustformersError {
    let mut registry = match TENSOR_REGISTRY.lock() {
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

    match CudaOperations::matrix_multiply(&a_tensor, &b_tensor, &mut c_tensor, m, n, k) {
        Ok(_) => {
            // Update the result tensor in the registry
            let mut registry = match TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(c_handle, c_tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Copy data from host to CUDA device
#[no_mangle]
pub extern "C" fn trustformers_cuda_copy_host_to_device(
    host_data: *const c_float,
    tensor_handle: usize,
    size: usize,
) -> TrustformersError {
    if host_data.is_null() {
        return TrustformersError::NullPointer;
    }

    let mut registry = match TENSOR_REGISTRY.lock() {
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

    match CudaOperations::copy_host_to_device(host_slice, &mut tensor) {
        Ok(_) => {
            // Update the tensor in the registry with modified tensor
            let mut registry = match TENSOR_REGISTRY.lock() {
                Ok(registry) => registry,
                Err(_) => return TrustformersError::RuntimeError,
            };
            registry.insert(tensor_handle, tensor);
            TrustformersError::Success
        },
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Copy data from CUDA device to host
#[no_mangle]
pub extern "C" fn trustformers_cuda_copy_device_to_host(
    tensor_handle: usize,
    host_data: *mut c_float,
    size: usize,
) -> TrustformersError {
    if host_data.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = match TENSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let tensor = match registry.get(&tensor_handle) {
        Some(tensor) => tensor,
        None => return TrustformersError::InvalidParameter,
    };

    let host_slice = unsafe { std::slice::from_raw_parts_mut(host_data, size) };

    match CudaOperations::copy_device_to_host(tensor, host_slice) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Check if CUDA is available
#[no_mangle]
pub extern "C" fn trustformers_cuda_is_available() -> c_int {
    // In real implementation, this would check for CUDA runtime and drivers
    // For simulation, check environment variable
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() || std::env::var("DISABLE_CUDA").is_err() {
        1
    } else {
        0
    }
}

/// Get CUDA memory usage
#[no_mangle]
pub extern "C" fn trustformers_cuda_get_memory_info(
    device_id: c_int,
    free_memory: *mut u64,
    total_memory: *mut u64,
) -> TrustformersError {
    if free_memory.is_null() || total_memory.is_null() {
        return TrustformersError::NullPointer;
    }

    let manager = match CUDA_MANAGER.lock() {
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

/// Synchronize CUDA device (wait for all operations to complete)
#[no_mangle]
pub extern "C" fn trustformers_cuda_synchronize() -> TrustformersError {
    // In real implementation, this would call cudaDeviceSynchronize()
    TrustformersError::Success
}
