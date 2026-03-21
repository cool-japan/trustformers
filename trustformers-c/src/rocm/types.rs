//! ROCm type definitions and implementations

use std::collections::HashMap;
use std::os::raw::c_void;

/// ROCm/HIP device information
#[derive(Debug, Clone)]
pub struct RocmDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_memory_mb: u64,
    pub free_memory_mb: u64,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory_per_block: u32,
    pub warp_size: i32,
    pub max_grid_size: [i32; 3],
    pub max_block_size: [i32; 3],
    pub pci_bus_id: String,
    pub pci_device_id: i32,
    pub clock_rate_khz: i32,
    pub memory_clock_rate_khz: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
}

/// ROCm context and device management
pub(crate) struct RocmManager {
    pub devices: Vec<RocmDeviceInfo>,
    pub current_device: Option<i32>,
    #[cfg(feature = "rocm")]
    pub hip_devices: HashMap<i32, HipDeviceHandle>,
    #[cfg(feature = "rocm")]
    pub rocblas_handles: HashMap<i32, RocblasHandle>,
    #[cfg(feature = "rocm")]
    pub hip_streams: HashMap<i32, Vec<HipStream>>,
    pub initialized: bool,
}

/// HIP device handle (ROCm-specific)
#[cfg(feature = "rocm")]
pub(crate) struct HipDeviceHandle {
    pub device_id: i32,
    pub device_ptr: *mut c_void,
}

#[cfg(feature = "rocm")]
unsafe impl Send for HipDeviceHandle {}
#[cfg(feature = "rocm")]
unsafe impl Sync for HipDeviceHandle {}

/// ROCblas handle wrapper
#[cfg(feature = "rocm")]
pub(crate) struct RocblasHandle {
    pub handle_ptr: *mut c_void,
    pub device_id: i32,
}

#[cfg(feature = "rocm")]
unsafe impl Send for RocblasHandle {}
#[cfg(feature = "rocm")]
unsafe impl Sync for RocblasHandle {}

/// HIP stream for async operations
#[cfg(feature = "rocm")]
pub(crate) struct HipStream {
    pub stream_ptr: *mut c_void,
    pub stream_id: usize,
    pub device_id: i32,
}

#[cfg(feature = "rocm")]
unsafe impl Send for HipStream {}
#[cfg(feature = "rocm")]
unsafe impl Sync for HipStream {}

impl std::fmt::Debug for RocmManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmManager")
            .field("devices", &self.devices)
            .field("current_device", &self.current_device)
            .field("initialized", &self.initialized)
            .finish()
    }
}

/// ROCm tensor allocation info
pub struct RocmTensor {
    #[cfg(feature = "rocm")]
    pub device_ptr: *mut c_void,
    #[cfg(not(feature = "rocm"))]
    pub device_ptr: usize,
    pub shape: Vec<usize>,
    pub dtype: TensorDataType,
    pub size_bytes: usize,
    pub device_id: i32,
}

impl std::fmt::Debug for RocmTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("size_bytes", &self.size_bytes)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl Clone for RocmTensor {
    fn clone(&self) -> Self {
        Self {
            device_ptr: self.device_ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
            size_bytes: self.size_bytes,
            device_id: self.device_id,
        }
    }
}

// SAFETY: RocmTensor can be safely sent between threads as it only holds
// a device pointer and metadata. The actual device memory is managed by
// the ROCm runtime, which is thread-safe.
#[cfg(feature = "rocm")]
unsafe impl Send for RocmTensor {}

#[cfg(feature = "rocm")]
unsafe impl Sync for RocmTensor {}

/// Tensor data types supported by ROCm backend
#[derive(Debug, Clone, Copy)]
pub enum TensorDataType {
    Float32,
    Float16,
    Int32,
    Int8,
    UInt8,
    Float64,
    Int16,
    Int64,
}

impl TensorDataType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            TensorDataType::Float32 => 4,
            TensorDataType::Float16 => 2,
            TensorDataType::Int32 => 4,
            TensorDataType::Int8 => 1,
            TensorDataType::UInt8 => 1,
            TensorDataType::Float64 => 8,
            TensorDataType::Int16 => 2,
            TensorDataType::Int64 => 8,
        }
    }

    #[cfg(feature = "rocm")]
    pub fn to_rocblas_datatype(&self) -> i32 {
        match self {
            TensorDataType::Float32 => 151, // rocblas_datatype_f32_r
            TensorDataType::Float16 => 150, // rocblas_datatype_f16_r
            TensorDataType::Int32 => 152,   // rocblas_datatype_i32_r
            TensorDataType::Int8 => 160,    // rocblas_datatype_i8_r
            TensorDataType::UInt8 => 161,   // rocblas_datatype_u8_r
            TensorDataType::Float64 => 153, // rocblas_datatype_f64_r
            TensorDataType::Int16 => 154,   // rocblas_datatype_i16_r
            TensorDataType::Int64 => 155,   // rocblas_datatype_i64_r
        }
    }
}

/// HIP device properties structure
#[cfg(feature = "rocm")]
#[derive(Default)]
pub(crate) struct HipDeviceProperties {
    pub name: String,
    pub major: i32,
    pub minor: i32,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub shared_mem_per_block: u32,
    pub warp_size: i32,
    pub max_grid_size: [i32; 3],
    pub max_threads_dim: [i32; 3],
    pub pci_bus_id: String,
    pub pci_device_id: i32,
    pub clock_rate: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
}
