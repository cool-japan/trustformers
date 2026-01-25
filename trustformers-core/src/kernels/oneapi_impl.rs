// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Intel oneAPI backend implementation for TrustformeRS
//!
//! This module provides integration with Intel's oneAPI unified programming model,
//! supporting DPC++ (SYCL), oneDNN, oneMKL, and Intel GPU/CPU optimization.

#![allow(dead_code)] // oneAPI backend implementation with FFI bindings
#![allow(unused_variables)] // Backend implementation with reserved parameters

use crate::errors::compute_error;
use crate::hardware::{DataType, HardwareCapabilities, HardwareMetrics, HardwareResult};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Intel oneAPI backend for unified CPU/GPU compute
#[derive(Debug)]
pub struct OneApiBackend {
    /// oneAPI context
    context: Arc<Mutex<OneApiContext>>,
    /// Backend configuration
    config: OneApiConfig,
    /// Kernel cache for compiled DPC++ kernels
    kernel_cache: HashMap<String, OneApiKernel>,
    /// Performance metrics
    metrics: Arc<Mutex<HardwareMetrics>>,
    /// Memory manager
    memory_manager: OneApiMemoryManager,
}

/// oneAPI execution context
#[derive(Debug)]
pub struct OneApiContext {
    /// SYCL queue for execution
    queue: *mut SyclQueue,
    /// Device selector
    device: OneApiDevice,
    /// Context handle
    context_handle: *mut SyclContext,
    /// Event pool for synchronization
    event_pool: Vec<*mut SyclEvent>,
}

// SAFETY: SYCL runtime handles are thread-safe internally
unsafe impl Send for OneApiContext {}
unsafe impl Sync for OneApiContext {}

/// oneAPI device representation
#[derive(Debug, Clone)]
pub struct OneApiDevice {
    /// Device type (CPU, GPU, FPGA)
    pub device_type: OneApiDeviceType,
    /// Device vendor
    pub vendor: String,
    /// Device name
    pub name: String,
    /// Compute units
    pub compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Global memory size
    pub global_memory_size: usize,
    /// Local memory size
    pub local_memory_size: usize,
    /// Device capabilities
    pub capabilities: OneApiCapabilities,
}

/// oneAPI device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OneApiDeviceType {
    /// Intel CPU (with AVX-512, AMX support)
    CPU,
    /// Intel GPU (Xe, Arc, Data Center GPU)
    GPU,
    /// Intel FPGA
    FPGA,
    /// Custom accelerator
    Custom,
}

/// oneAPI device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneApiCapabilities {
    /// Supports double precision
    pub supports_fp64: bool,
    /// Supports half precision
    pub supports_fp16: bool,
    /// Supports Intel AMX (Advanced Matrix Extensions)
    pub supports_amx: bool,
    /// Supports AVX-512
    pub supports_avx512: bool,
    /// Supports Intel DL Boost
    pub supports_dl_boost: bool,
    /// Supports unified shared memory
    pub supports_usm: bool,
    /// Maximum allocation size
    pub max_allocation_size: usize,
    /// Preferred vector width
    pub preferred_vector_width: u32,
}

/// oneAPI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneApiConfig {
    /// Target device type
    pub device_type: OneApiDeviceType,
    /// Device selector preference
    pub device_preference: DevicePreference,
    /// Enable Intel oneDNN optimization
    pub enable_onednn: bool,
    /// Enable Intel oneMKL
    pub enable_onemkl: bool,
    /// Enable unified shared memory
    pub enable_usm: bool,
    /// Work group size optimization
    pub work_group_size: Option<usize>,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
    /// Custom oneAPI options
    pub custom_options: HashMap<String, String>,
}

/// Device selection preference
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DevicePreference {
    /// Prefer CPU execution
    CPU,
    /// Prefer GPU execution
    GPU,
    /// Automatic selection based on workload
    Auto,
    /// Use highest performance device
    HighestPerformance,
    /// Use lowest power consumption device
    LowestPower,
}

/// Memory optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MemoryOptimization {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Aggressive optimization
    Aggressive,
    /// Custom optimization
    Custom,
}

/// Compiled oneAPI kernel
#[derive(Debug, Clone)]
pub struct OneApiKernel {
    /// Kernel name
    name: String,
    /// Compiled kernel handle
    kernel_handle: *mut SyclKernel,
    /// Source code
    source: String,
    /// Compilation metadata
    metadata: OneApiCompilationMetadata,
    /// Kernel arguments specification
    arg_specs: Vec<KernelArgSpec>,
}

/// Kernel argument specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelArgSpec {
    /// Argument index
    pub index: usize,
    /// Argument name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Memory access pattern
    pub access_pattern: MemoryAccessPattern,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Memory access patterns for optimization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MemoryAccessPattern {
    /// Read-only access
    ReadOnly,
    /// Write-only access
    WriteOnly,
    /// Read-write access
    ReadWrite,
    /// Random access
    RandomAccess,
    /// Sequential access
    SequentialAccess,
    /// Coalesced access
    CoalescedAccess,
}

/// oneAPI compilation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneApiCompilationMetadata {
    /// Compilation time in milliseconds
    pub compilation_time_ms: f64,
    /// Binary size in bytes
    pub binary_size_bytes: usize,
    /// Optimization level
    pub optimization_level: u32,
    /// Target device
    pub target_device: OneApiDeviceType,
    /// Optimizations applied
    pub optimizations: Vec<String>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Kernel resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Register usage
    pub registers_used: u32,
    /// Shared memory usage in bytes
    pub shared_memory_bytes: usize,
    /// Private memory usage in bytes
    pub private_memory_bytes: usize,
    /// Work group size limits
    pub work_group_size_limits: (usize, usize, usize),
}

/// oneAPI memory manager
#[derive(Debug)]
pub struct OneApiMemoryManager {
    /// Available memory pools
    memory_pools: HashMap<String, MemoryPool>,
    /// Unified shared memory allocations
    usm_allocations: HashMap<String, UsmAllocation>,
    /// Memory optimization strategy
    optimization_strategy: MemoryOptimization,
}

/// Memory pool for different allocation types
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool name
    pub name: String,
    /// Pool type
    pub pool_type: MemoryPoolType,
    /// Total size in bytes
    pub total_size: usize,
    /// Used size in bytes
    pub used_size: usize,
    /// Pool handle
    pub handle: *mut MemoryPoolHandle,
}

/// Memory pool types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryPoolType {
    /// Device memory
    Device,
    /// Host memory
    Host,
    /// Shared memory
    Shared,
    /// Unified shared memory
    USM,
}

/// Unified shared memory allocation
#[derive(Debug, Clone)]
pub struct UsmAllocation {
    /// Allocation ID
    pub id: String,
    /// Memory pointer
    pub ptr: *mut u8,
    /// Size in bytes
    pub size: usize,
    /// USM type
    pub usm_type: UsmType,
    /// Allocated timestamp
    pub allocated_at: Instant,
}

/// USM (Unified Shared Memory) types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UsmType {
    /// Device USM
    Device,
    /// Host USM
    Host,
    /// Shared USM
    Shared,
}

// Foreign function interface for Intel oneAPI runtime
extern "C" {
    // SYCL Queue management
    fn sycl_queue_create(device_type: i32, device_id: i32) -> *mut SyclQueue;
    fn sycl_queue_destroy(queue: *mut SyclQueue);
    fn sycl_queue_submit(
        queue: *mut SyclQueue,
        kernel: *mut SyclKernel,
        global_size: *const usize,
        local_size: *const usize,
    ) -> *mut SyclEvent;
    fn sycl_queue_wait(queue: *mut SyclQueue) -> i32;

    // Kernel compilation and execution
    fn sycl_kernel_compile(
        source: *const i8,
        source_len: usize,
        options: *const i8,
    ) -> *mut SyclKernel;
    fn sycl_kernel_destroy(kernel: *mut SyclKernel);
    fn sycl_kernel_set_arg(kernel: *mut SyclKernel, index: u32, arg: *const u8, size: usize)
        -> i32;

    // Memory management
    fn sycl_malloc_device(size: usize, queue: *mut SyclQueue) -> *mut u8;
    fn sycl_malloc_host(size: usize, queue: *mut SyclQueue) -> *mut u8;
    fn sycl_malloc_shared(size: usize, queue: *mut SyclQueue) -> *mut u8;
    fn sycl_free(ptr: *mut u8, queue: *mut SyclQueue);
    fn sycl_memcpy(
        dst: *mut u8,
        src: *const u8,
        size: usize,
        queue: *mut SyclQueue,
    ) -> *mut SyclEvent;

    // Device information
    fn sycl_get_device_count(device_type: i32) -> i32;
    fn sycl_get_device_info(device_type: i32, device_id: i32, info: *mut DeviceInfo) -> i32;

    // oneDNN integration
    fn onednn_init() -> i32;
    fn onednn_create_convolution(
        src_desc: *const TensorDesc,
        weights_desc: *const TensorDesc,
        dst_desc: *const TensorDesc,
    ) -> *mut OneDnnOp;
    fn onednn_execute(op: *mut OneDnnOp, inputs: *const *const f32, outputs: *mut *mut f32) -> i32;

    // oneMKL integration
    fn onemkl_init() -> i32;
    fn onemkl_gemm(
        queue: *mut SyclQueue,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
    ) -> i32;
    fn onemkl_conv2d(
        queue: *mut SyclQueue,
        input: *const f32,
        kernel: *const f32,
        output: *mut f32,
        params: *const ConvParams,
    ) -> i32;
}

// Opaque handle types for FFI
#[repr(C)]
pub struct SyclQueue {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SyclKernel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SyclEvent {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SyclContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OneDnnOp {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MemoryPoolHandle {
    _private: [u8; 0],
}

/// Device information structure for FFI
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: i32,
    pub vendor_id: u32,
    pub device_name: [i8; 256],
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub global_memory_size: u64,
    pub local_memory_size: u64,
    pub supports_fp64: i32,
    pub supports_fp16: i32,
}

/// Tensor descriptor for oneDNN
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub dims: [i32; 8],
    pub ndims: i32,
    pub data_type: i32,
    pub format: i32,
}

/// Convolution parameters for oneMKL
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ConvParams {
    pub input_dims: [i32; 4],
    pub kernel_dims: [i32; 4],
    pub output_dims: [i32; 4],
    pub strides: [i32; 2],
    pub padding: [i32; 2],
}

impl OneApiBackend {
    /// Create a new Intel oneAPI backend
    pub fn new(config: OneApiConfig) -> HardwareResult<Self> {
        let context = Arc::new(Mutex::new(Self::initialize_context(&config)?));

        let metrics = Arc::new(Mutex::new(HardwareMetrics {
            ops_per_second: 0.0,
            memory_bandwidth: Self::get_memory_bandwidth(&config.device_type),
            utilization: 0.0,
            power_consumption: 0.0,
            temperature: None,
            error_rate: 0.0,
            latency: 0.0,
            throughput: 0.0,
        }));

        let memory_manager = OneApiMemoryManager::new(config.memory_optimization);

        // Initialize oneDNN and oneMKL if enabled
        if config.enable_onednn {
            unsafe {
                let result = onednn_init();
                if result != 0 {
                    eprintln!("Warning: oneDNN initialization failed");
                }
            }
        }

        if config.enable_onemkl {
            unsafe {
                let result = onemkl_init();
                if result != 0 {
                    eprintln!("Warning: oneMKL initialization failed");
                }
            }
        }

        Ok(Self {
            context,
            config,
            kernel_cache: HashMap::new(),
            metrics,
            memory_manager,
        })
    }

    /// Compile a DPC++ kernel
    pub fn compile_kernel(
        &mut self,
        name: &str,
        source: &str,
        arg_specs: &[KernelArgSpec],
    ) -> HardwareResult<String> {
        let kernel_id = format!("{}_{}", name, arg_specs.len());

        if self.kernel_cache.contains_key(&kernel_id) {
            return Ok(kernel_id);
        }

        let start_time = Instant::now();

        let source_cstring = CString::new(source)
            .map_err(|_| compute_error("oneapi_operation", "Invalid kernel source"))?;

        let options = self.get_compilation_options();
        let options_cstring = CString::new(options)
            .map_err(|_| compute_error("oneapi_operation", "Invalid compilation options"))?;

        let kernel_handle = unsafe {
            sycl_kernel_compile(
                source_cstring.as_ptr(),
                source.len(),
                options_cstring.as_ptr(),
            )
        };

        if kernel_handle.is_null() {
            return Err(compute_error(
                "oneapi_operation",
                "Kernel compilation failed",
            ));
        }

        let compilation_time = start_time.elapsed().as_millis() as f64;

        let metadata = OneApiCompilationMetadata {
            compilation_time_ms: compilation_time,
            binary_size_bytes: source.len(),
            optimization_level: 3,
            target_device: self.config.device_type,
            optimizations: self.get_applied_optimizations(),
            resource_usage: ResourceUsage {
                registers_used: 32,         // Estimated
                shared_memory_bytes: 1024,  // Estimated
                private_memory_bytes: 2048, // Estimated
                work_group_size_limits: (256, 256, 64),
            },
        };

        let kernel = OneApiKernel {
            name: name.to_string(),
            kernel_handle,
            source: source.to_string(),
            metadata,
            arg_specs: arg_specs.to_vec(),
        };

        self.kernel_cache.insert(kernel_id.clone(), kernel);
        Ok(kernel_id)
    }

    /// Execute a compiled kernel
    pub fn execute_kernel(
        &mut self,
        kernel_id: &str,
        inputs: &[Tensor],
        global_size: &[usize],
        local_size: Option<&[usize]>,
    ) -> HardwareResult<Vec<Tensor>> {
        let kernel = self
            .kernel_cache
            .get(kernel_id)
            .ok_or_else(|| compute_error("oneapi_operation", "Kernel not found"))?;

        let start_time = Instant::now();

        // Set kernel arguments
        for (i, input) in inputs.iter().enumerate() {
            let result = unsafe {
                sycl_kernel_set_arg(
                    kernel.kernel_handle,
                    i as u32,
                    input.data()?.as_ptr() as *const u8,
                    input.size_bytes(),
                )
            };

            if result != 0 {
                return Err(compute_error(
                    "oneapi_operation",
                    "Failed to set kernel argument",
                ));
            }
        }

        // Execute kernel
        {
            let context = self.context.lock().expect("Lock poisoned");
            let local_ptr = local_size.map(|ls| ls.as_ptr()).unwrap_or(std::ptr::null());

            let event = unsafe {
                sycl_queue_submit(
                    context.queue,
                    kernel.kernel_handle,
                    global_size.as_ptr(),
                    local_ptr,
                )
            };

            if event.is_null() {
                return Err(compute_error("oneapi_operation", "Kernel execution failed"));
            }

            // Wait for completion
            let result = unsafe { sycl_queue_wait(context.queue) };
            if result != 0 {
                return Err(compute_error(
                    "oneapi_operation",
                    "Kernel execution wait failed",
                ));
            }
        } // context lock is dropped here

        // Create output tensors (simplified - in practice would need proper output handling)
        let output_tensors = self.create_output_tensors(inputs)?;

        // Update metrics
        let execution_time = start_time.elapsed();
        let metadata = kernel.metadata.clone();
        self.update_execution_metrics(execution_time, &metadata);

        Ok(output_tensors)
    }

    /// Execute oneDNN convolution operation
    pub fn execute_onednn_conv2d(
        &mut self,
        input: &Tensor,
        weights: &Tensor,
        bias: Option<&Tensor>,
        strides: &[usize],
        padding: &[usize],
    ) -> HardwareResult<Tensor> {
        if !self.config.enable_onednn {
            return Err(compute_error("oneapi_operation", "oneDNN not enabled"));
        }

        let input_desc = self.tensor_to_onednn_desc(input);
        let weights_desc = self.tensor_to_onednn_desc(weights);
        let output_shape =
            self.compute_conv_output_shape(&input.shape(), &weights.shape(), strides, padding);
        let output_desc = TensorDesc {
            dims: [
                output_shape[0] as i32,
                output_shape[1] as i32,
                output_shape[2] as i32,
                output_shape[3] as i32,
                0,
                0,
                0,
                0,
            ],
            ndims: 4,
            data_type: 0, // Float32
            format: 0,    // NCHW
        };

        let conv_op =
            unsafe { onednn_create_convolution(&input_desc, &weights_desc, &output_desc) };

        if conv_op.is_null() {
            return Err(compute_error(
                "oneapi_operation",
                "Failed to create oneDNN convolution",
            ));
        }

        let mut output_data = vec![0.0f32; output_shape.iter().product()];
        let input_data = input.data()?;
        let inputs = [input_data.as_ptr()];
        let mut outputs = [output_data.as_mut_ptr()];

        let result = unsafe { onednn_execute(conv_op, inputs.as_ptr(), outputs.as_mut_ptr()) };

        if result != 0 {
            return Err(compute_error(
                "oneapi_operation",
                "oneDNN convolution execution failed",
            ));
        }

        Tensor::from_vec(output_data, &output_shape)
    }

    /// Execute oneMKL GEMM operation
    pub fn execute_onemkl_gemm(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        c: Option<&Tensor>,
    ) -> HardwareResult<Tensor> {
        if !self.config.enable_onemkl {
            return Err(compute_error("oneapi_operation", "oneMKL not enabled"));
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
            return Err(compute_error(
                "oneapi_operation",
                "Invalid matrix dimensions for GEMM",
            ));
        }

        let m = a_shape[0] as i32;
        let n = b_shape[1] as i32;
        let k = a_shape[1] as i32;

        let output_shape = vec![a_shape[0], b_shape[1]];
        let mut output_data = if let Some(c_tensor) = c {
            c_tensor.data()?.clone()
        } else {
            vec![0.0f32; output_shape.iter().product()]
        };

        let context = self.context.lock().expect("Lock poisoned");
        let result = unsafe {
            onemkl_gemm(
                context.queue,
                m,
                n,
                k,
                a.data()?.as_ptr(),
                b.data()?.as_ptr(),
                output_data.as_mut_ptr(),
            )
        };

        if result != 0 {
            return Err(compute_error(
                "oneapi_operation",
                "oneMKL GEMM execution failed",
            ));
        }

        Tensor::from_vec(output_data, &output_shape)
    }

    /// Get backend capabilities
    pub fn get_capabilities(&self) -> HardwareCapabilities {
        let data_types = match self.config.device_type {
            OneApiDeviceType::CPU => vec![
                DataType::F32,
                DataType::F64,
                DataType::I32,
                DataType::I64,
                DataType::I16,
                DataType::I8,
                DataType::Bool,
            ],
            OneApiDeviceType::GPU => vec![
                DataType::F32,
                DataType::F16,
                DataType::I32,
                DataType::I16,
                DataType::I8,
                DataType::Bool,
            ],
            OneApiDeviceType::FPGA => {
                vec![DataType::F32, DataType::I32, DataType::I16, DataType::I8]
            },
            OneApiDeviceType::Custom => vec![DataType::F32, DataType::I32],
        };

        let (compute_units, memory_size, power_consumption) = match self.config.device_type {
            OneApiDeviceType::CPU => (16, 64 * 1024 * 1024 * 1024, 125.0), // 16 cores, 64GB, 125W
            OneApiDeviceType::GPU => (96, 16 * 1024 * 1024 * 1024, 225.0), // 96 EUs, 16GB, 225W
            OneApiDeviceType::FPGA => (1, 8 * 1024 * 1024 * 1024, 75.0),   // 1 device, 8GB, 75W
            OneApiDeviceType::Custom => (8, 8 * 1024 * 1024 * 1024, 100.0), // 8 units, 8GB, 100W
        };

        HardwareCapabilities {
            data_types,
            max_dimensions: 8,
            memory_size: Some(memory_size),
            clock_frequency: Some(match self.config.device_type {
                OneApiDeviceType::CPU => 3_200_000_000,    // 3.2 GHz
                OneApiDeviceType::GPU => 2_100_000_000,    // 2.1 GHz
                OneApiDeviceType::FPGA => 300_000_000,     // 300 MHz
                OneApiDeviceType::Custom => 1_000_000_000, // 1 GHz
            }),
            compute_units: Some(compute_units),
            operations: vec![
                "gemm".to_string(),
                "conv2d".to_string(),
                "batch_norm".to_string(),
                "activation".to_string(),
                "pooling".to_string(),
                "attention".to_string(),
                "layer_norm".to_string(),
                "softmax".to_string(),
                "reduce".to_string(),
                "transpose".to_string(),
                "reshape".to_string(),
            ],
            power_consumption: Some(power_consumption),
            thermal_design_power: Some(power_consumption * 1.3), // 30% overhead
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> HardwareMetrics {
        self.metrics.lock().expect("Lock poisoned").clone()
    }

    // Private helper methods
    fn initialize_context(config: &OneApiConfig) -> HardwareResult<OneApiContext> {
        let device_type_id = match config.device_type {
            OneApiDeviceType::CPU => 0,
            OneApiDeviceType::GPU => 1,
            OneApiDeviceType::FPGA => 2,
            OneApiDeviceType::Custom => 3,
        };

        let queue = unsafe { sycl_queue_create(device_type_id, 0) };
        if queue.is_null() {
            return Err(compute_error(
                "oneapi_operation",
                "Failed to create SYCL queue",
            ));
        }

        let device = Self::get_device_info(config.device_type)?;

        Ok(OneApiContext {
            queue,
            device,
            context_handle: std::ptr::null_mut(), // Simplified
            event_pool: Vec::new(),
        })
    }

    fn get_device_info(device_type: OneApiDeviceType) -> HardwareResult<OneApiDevice> {
        let device_type_id = match device_type {
            OneApiDeviceType::CPU => 0,
            OneApiDeviceType::GPU => 1,
            OneApiDeviceType::FPGA => 2,
            OneApiDeviceType::Custom => 3,
        };

        let mut info = DeviceInfo {
            device_type: device_type_id,
            vendor_id: 0x8086, // Intel
            device_name: [0; 256],
            compute_units: 0,
            max_work_group_size: 0,
            global_memory_size: 0,
            local_memory_size: 0,
            supports_fp64: 0,
            supports_fp16: 0,
        };

        let result = unsafe { sycl_get_device_info(device_type_id, 0, &mut info) };
        if result != 0 {
            return Err(compute_error(
                "oneapi_operation",
                "Failed to get device info",
            ));
        }

        let device_name =
            unsafe { CStr::from_ptr(info.device_name.as_ptr()).to_string_lossy().to_string() };

        Ok(OneApiDevice {
            device_type,
            vendor: "Intel".to_string(),
            name: device_name,
            compute_units: info.compute_units,
            max_work_group_size: info.max_work_group_size,
            global_memory_size: info.global_memory_size as usize,
            local_memory_size: info.local_memory_size as usize,
            capabilities: OneApiCapabilities {
                supports_fp64: info.supports_fp64 != 0,
                supports_fp16: info.supports_fp16 != 0,
                supports_amx: device_type == OneApiDeviceType::CPU,
                supports_avx512: device_type == OneApiDeviceType::CPU,
                supports_dl_boost: true,
                supports_usm: true,
                max_allocation_size: info.global_memory_size as usize / 4,
                preferred_vector_width: match device_type {
                    OneApiDeviceType::CPU => 16, // AVX-512
                    OneApiDeviceType::GPU => 8,  // SIMD8
                    OneApiDeviceType::FPGA => 4, // Custom
                    OneApiDeviceType::Custom => 8,
                },
            },
        })
    }

    fn get_memory_bandwidth(device_type: &OneApiDeviceType) -> f64 {
        match device_type {
            OneApiDeviceType::CPU => 100e9,    // 100 GB/s
            OneApiDeviceType::GPU => 500e9,    // 500 GB/s
            OneApiDeviceType::FPGA => 50e9,    // 50 GB/s
            OneApiDeviceType::Custom => 200e9, // 200 GB/s
        }
    }

    fn get_compilation_options(&self) -> String {
        let mut options = vec!["-O3"];

        if self.config.device_type == OneApiDeviceType::CPU {
            options.push("-march=native");
            options.push("-mavx512f");
        }

        if self.config.enable_usm {
            options.push("-fsycl-unnamed-lambda");
        }

        options.join(" ")
    }

    fn get_applied_optimizations(&self) -> Vec<String> {
        let mut optimizations = vec![
            "loop_unrolling".to_string(),
            "vectorization".to_string(),
            "memory_coalescing".to_string(),
        ];

        match self.config.device_type {
            OneApiDeviceType::CPU => {
                optimizations.extend(vec![
                    "avx512_optimization".to_string(),
                    "cache_blocking".to_string(),
                    "amx_optimization".to_string(),
                ]);
            },
            OneApiDeviceType::GPU => {
                optimizations.extend(vec![
                    "simd_optimization".to_string(),
                    "work_group_optimization".to_string(),
                    "barrier_elimination".to_string(),
                ]);
            },
            OneApiDeviceType::FPGA => {
                optimizations.extend(vec![
                    "pipeline_optimization".to_string(),
                    "resource_sharing".to_string(),
                ]);
            },
            OneApiDeviceType::Custom => {
                optimizations.push("custom_optimization".to_string());
            },
        }

        optimizations
    }

    fn create_output_tensors(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        // Simplified - in practice would need proper output shape inference
        let output_shape = inputs[0].shape().to_vec();
        let output_data = vec![0.0f32; output_shape.iter().product()];
        let output_tensor = Tensor::from_vec(output_data, &output_shape)?;
        Ok(vec![output_tensor])
    }

    fn tensor_to_onednn_desc(&self, tensor: &Tensor) -> TensorDesc {
        let shape = tensor.shape();
        let mut dims = [0i32; 8];
        for (i, &dim) in shape.iter().take(8).enumerate() {
            dims[i] = dim as i32;
        }

        TensorDesc {
            dims,
            ndims: shape.len() as i32,
            data_type: 0, // Float32
            format: 0,    // Default format
        }
    }

    fn compute_conv_output_shape(
        &self,
        input_shape: &[usize],
        kernel_shape: &[usize],
        strides: &[usize],
        padding: &[usize],
    ) -> Vec<usize> {
        vec![
            input_shape[0],                                                       // batch size
            kernel_shape[0],                                                      // output channels
            (input_shape[2] + 2 * padding[0] - kernel_shape[2]) / strides[0] + 1, // height
            (input_shape[3] + 2 * padding[1] - kernel_shape[3]) / strides[1] + 1, // width
        ]
    }

    fn update_execution_metrics(
        &mut self,
        execution_time: Duration,
        metadata: &OneApiCompilationMetadata,
    ) {
        let mut metrics = self.metrics.lock().expect("Lock poisoned");
        let execution_ms = execution_time.as_millis() as f64;

        // Simplified metrics update
        metrics.latency = execution_ms;
        metrics.throughput = 1000.0 / execution_ms; // Operations per second
        metrics.utilization = 0.8; // Estimated utilization
    }
}

impl OneApiMemoryManager {
    fn new(optimization: MemoryOptimization) -> Self {
        Self {
            memory_pools: HashMap::new(),
            usm_allocations: HashMap::new(),
            optimization_strategy: optimization,
        }
    }

    /// Allocate unified shared memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The `queue` pointer is valid and points to an initialized SYCL queue
    /// - The queue remains valid for the lifetime of the allocation
    /// - The returned pointer is not used after deallocating via `deallocate_usm`
    pub unsafe fn allocate_usm(
        &mut self,
        id: String,
        size: usize,
        usm_type: UsmType,
        queue: *mut SyclQueue,
    ) -> HardwareResult<*mut u8> {
        let ptr = unsafe {
            match usm_type {
                UsmType::Device => sycl_malloc_device(size, queue),
                UsmType::Host => sycl_malloc_host(size, queue),
                UsmType::Shared => sycl_malloc_shared(size, queue),
            }
        };

        if ptr.is_null() {
            return Err(compute_error("oneapi_operation", "USM allocation failed"));
        }

        let allocation = UsmAllocation {
            id: id.clone(),
            ptr,
            size,
            usm_type,
            allocated_at: Instant::now(),
        };

        self.usm_allocations.insert(id, allocation);
        Ok(ptr)
    }

    /// Deallocate unified shared memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The `queue` pointer is valid and points to an initialized SYCL queue
    /// - The allocation identified by `id` was previously allocated via `allocate_usm`
    /// - No references to the allocated memory exist after this call
    pub unsafe fn deallocate_usm(&mut self, id: &str, queue: *mut SyclQueue) -> HardwareResult<()> {
        if let Some(allocation) = self.usm_allocations.remove(id) {
            unsafe {
                sycl_free(allocation.ptr, queue);
            }
            Ok(())
        } else {
            Err(compute_error(
                "oneapi_operation",
                "USM allocation not found",
            ))
        }
    }
}

impl Default for OneApiConfig {
    fn default() -> Self {
        Self {
            device_type: OneApiDeviceType::CPU,
            device_preference: DevicePreference::Auto,
            enable_onednn: true,
            enable_onemkl: true,
            enable_usm: true,
            work_group_size: None,
            memory_optimization: MemoryOptimization::Basic,
            custom_options: HashMap::new(),
        }
    }
}

impl Drop for OneApiContext {
    fn drop(&mut self) {
        if !self.queue.is_null() {
            unsafe {
                sycl_queue_destroy(self.queue);
            }
        }
    }
}

/// Utility functions for oneAPI integration
pub mod utils {
    use super::*;

    /// Check if Intel oneAPI is available
    pub fn is_oneapi_available() -> bool {
        let cpu_count = unsafe { sycl_get_device_count(0) };
        let gpu_count = unsafe { sycl_get_device_count(1) };
        cpu_count > 0 || gpu_count > 0
    }

    /// Get available oneAPI devices
    pub fn get_available_devices() -> Vec<OneApiDevice> {
        let mut devices = Vec::new();

        // Check CPU devices
        let cpu_count = unsafe { sycl_get_device_count(0) };
        for i in 0..cpu_count {
            if let Ok(device) = OneApiBackend::get_device_info(OneApiDeviceType::CPU) {
                devices.push(device);
            }
        }

        // Check GPU devices
        let gpu_count = unsafe { sycl_get_device_count(1) };
        for i in 0..gpu_count {
            if let Ok(device) = OneApiBackend::get_device_info(OneApiDeviceType::GPU) {
                devices.push(device);
            }
        }

        devices
    }

    /// Generate optimized DPC++ kernel for matrix multiplication
    pub fn generate_gemm_kernel(m: usize, n: usize, k: usize) -> String {
        r#"
#include <sycl/sycl.hpp>

class GemmKernel;

void gemm_kernel(sycl::queue& q, const float* A, const float* B, float* C,
                 int M, int N, int K) {
    auto range = sycl::range<2>(M, N);
    auto local_range = sycl::range<2>(16, 16);

    q.parallel_for<GemmKernel>(
        sycl::nd_range<2>(range, local_range),
        [=](sycl::nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);

            if (row < M && col < N) {
                float sum = 0.0f;
                for (int i = 0; i < K; ++i) {
                    sum += A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    ).wait();
}
"#
        .to_string()
    }

    /// Generate optimized DPC++ kernel for convolution
    pub fn generate_conv2d_kernel(
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
    ) -> String {
        r#"
#include <sycl/sycl.hpp>

class Conv2dKernel;

void conv2d_kernel(sycl::queue& q, const float* input, const float* weights,
                   float* output, int batch, int in_channels, int out_channels,
                   int height, int width, int kernel_size) {
    auto range = sycl::range<3>(batch * out_channels, height, width);
    auto local_range = sycl::range<3>(1, 16, 16);

    q.parallel_for<Conv2dKernel>(
        sycl::nd_range<3>(range, local_range),
        [=](sycl::nd_item<3> item) {
            int b_oc = item.get_global_id(0);
            int h = item.get_global_id(1);
            int w = item.get_global_id(2);

            int b = b_oc / out_channels;
            int oc = b_oc % out_channels;

            if (b < batch && h < height && w < width) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = h + kh;
                            int iw = w + kw;
                            if (ih < height + kernel_size - 1 && iw < width + kernel_size - 1) {
                                sum += input[((b * in_channels + ic) * (height + kernel_size - 1) + ih) * (width + kernel_size - 1) + iw] *
                                       weights[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                            }
                        }
                    }
                }
                output[((b * out_channels + oc) * height + h) * width + w] = sum;
            }
        }
    ).wait();
}
"#.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oneapi_device_type_serialization() {
        let device_type = OneApiDeviceType::GPU;
        let serialized = serde_json::to_string(&device_type).expect("JSON serialization failed");
        let deserialized: OneApiDeviceType =
            serde_json::from_str(&serialized).expect("JSON deserialization failed");
        assert_eq!(device_type, deserialized);
    }

    #[test]
    fn test_oneapi_config_default() {
        let config = OneApiConfig::default();
        assert_eq!(config.device_type, OneApiDeviceType::CPU);
        assert_eq!(config.device_preference, DevicePreference::Auto);
        assert!(config.enable_onednn);
        assert!(config.enable_onemkl);
    }

    #[test]
    fn test_memory_access_patterns() {
        let patterns = [
            MemoryAccessPattern::ReadOnly,
            MemoryAccessPattern::WriteOnly,
            MemoryAccessPattern::ReadWrite,
            MemoryAccessPattern::CoalescedAccess,
        ];
        assert_eq!(patterns.len(), 4);
        assert_eq!(patterns[0], MemoryAccessPattern::ReadOnly);
    }

    #[test]
    fn test_usm_types() {
        let usm_types = [UsmType::Device, UsmType::Host, UsmType::Shared];
        assert_eq!(usm_types.len(), 3);
        assert_eq!(usm_types[0], UsmType::Device);
        assert_eq!(usm_types[2], UsmType::Shared);
    }

    #[test]
    fn test_kernel_generation() {
        let gemm_kernel = utils::generate_gemm_kernel(128, 128, 128);
        assert!(gemm_kernel.contains("GemmKernel"));
        assert!(gemm_kernel.contains("parallel_for"));

        let conv_kernel = utils::generate_conv2d_kernel(64, 128, 3);
        assert!(conv_kernel.contains("Conv2dKernel"));
        assert!(conv_kernel.contains("nd_range<3>"));
    }
}
