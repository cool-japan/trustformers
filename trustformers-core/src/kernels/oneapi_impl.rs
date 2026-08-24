// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Intel oneAPI backend implementation for TrustformeRS
//!
//! This module models the context/queue/kernel-cache API surface (DPC++
//! (SYCL) queues, oneDNN, oneMKL, Intel GPU/CPU device selection) that a
//! real integration with Intel's oneAPI unified programming model would
//! expose. It does **not** link or execute a real oneAPI runtime:
//! `--features oneapi` has zero dependencies (`oneapi = []` in
//! `Cargo.toml`), the `extern "C"` FFI declarations further down are
//! permanently disabled with `cfg(any())`, and every public method that
//! would need actual SYCL/oneDNN/oneMKL execution returns a structured
//! [`HardwareResult`] error naming exactly what is missing ("no
//! SYCL/oneDNN/oneMKL runtime is linked") instead of fabricating output
//! tensors, device counts, or timings. See the `HONESTY NOTE` above the
//! `extern "C"` block for the full rationale and what wiring a real backend
//! would require.

#![allow(dead_code)] // oneAPI backend implementation with FFI bindings

use crate::errors::compute_error;
use crate::hardware::{DataType, HardwareCapabilities, HardwareMetrics, HardwareResult};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

// Foreign function interface for Intel oneAPI runtime.
//
// HONESTY NOTE: these symbols are declared but nothing in this workspace
// provides them - there is no `-sys` crate, no `#[link(name = "...")]`
// attribute, and `build.rs` links only `framework=Accelerate` on macOS.
// `--features oneapi` has zero dependencies (`oneapi = []` in Cargo.toml),
// so any real DPC++/SYCL runtime call here would be an unresolved symbol at
// final link time. This block (and every call site below) is therefore
// gated behind `cfg(any())` - permanently disabled - so the crate never
// tries to link against a runtime that was never provided. Every public
// method that used to call into it now returns a structured
// `NotSupported`/hardware-unavailable error instead of silently returning
// fabricated data. Wiring a real backend means adding a genuine `-sys`
// binding crate, a `#[link(...)]` target, and a `build.rs` probe, then
// replacing the `cfg(any())` gate below with a real feature check.
#[cfg(any())]
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
    /// Create a new Intel oneAPI backend.
    ///
    /// Always returns `Err`: this build has no real SYCL/oneDNN/oneMKL
    /// runtime linked (see the `extern "C"` block above), so there is no
    /// honest way to construct a working backend. This replaces what used
    /// to be a "successful" construction backed entirely by unresolved FFI
    /// calls and fabricated device data.
    pub fn new(_config: OneApiConfig) -> HardwareResult<Self> {
        Err(compute_error(
            "oneapi_operation",
            "Intel oneAPI backend is not available: no SYCL/oneDNN/oneMKL runtime is linked \
             into this build (the `oneapi` feature has no real backend binding yet)",
        ))
    }

    /// Compile a DPC++ kernel.
    ///
    /// Always errors: no SYCL compiler is linked into this build (see the
    /// `extern "C"` block above). A `OneApiBackend` can never actually be
    /// constructed (`new` always errors), so this is unreachable from safe
    /// code; the honest error is here in case that ever changes.
    pub fn compile_kernel(
        &mut self,
        _name: &str,
        _source: &str,
        _arg_specs: &[KernelArgSpec],
    ) -> HardwareResult<String> {
        Err(compute_error(
            "oneapi_operation",
            "Intel oneAPI kernel compilation is not available: no SYCL runtime is linked",
        ))
    }

    /// Execute a compiled kernel. Always errors (see `compile_kernel`).
    pub fn execute_kernel(
        &mut self,
        _kernel_id: &str,
        _inputs: &[Tensor],
        _global_size: &[usize],
        _local_size: Option<&[usize]>,
    ) -> HardwareResult<Vec<Tensor>> {
        Err(compute_error(
            "oneapi_operation",
            "Intel oneAPI kernel execution is not available: no SYCL runtime is linked",
        ))
    }

    /// Execute oneDNN convolution operation. Always errors (see
    /// `compile_kernel`): no oneDNN runtime is linked into this build.
    pub fn execute_onednn_conv2d(
        &mut self,
        _input: &Tensor,
        _weights: &Tensor,
        _bias: Option<&Tensor>,
        _strides: &[usize],
        _padding: &[usize],
    ) -> HardwareResult<Tensor> {
        Err(compute_error(
            "oneapi_operation",
            "oneDNN convolution is not available: no oneDNN runtime is linked",
        ))
    }

    /// Execute oneMKL GEMM operation. Always errors (see `compile_kernel`):
    /// no oneMKL runtime is linked into this build.
    pub fn execute_onemkl_gemm(
        &mut self,
        _a: &Tensor,
        _b: &Tensor,
        _c: Option<&Tensor>,
    ) -> HardwareResult<Tensor> {
        Err(compute_error(
            "oneapi_operation",
            "oneMKL GEMM is not available: no oneMKL runtime is linked",
        ))
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
        self.metrics.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    // Private helper methods.
    //
    // `initialize_context`/`get_device_info` (which used to call
    // `sycl_queue_create`/`sycl_get_device_info` and, on top of that,
    // fabricated most of `OneApiDevice`'s fields even on a "successful"
    // call - e.g. `supports_dl_boost: true` unconditionally) are deleted
    // rather than kept as dead code: nothing constructs a real
    // `OneApiContext`/`OneApiDevice` anymore (see `OneApiBackend::new` and
    // `utils::get_available_devices`), and keeping them around would only
    // invite a future caller to wire them back up as real device data
    // when they are not.
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

    /// Historically returned `inputs[0].shape()` filled with zeros and
    /// called it the kernel's output - i.e. `execute_kernel` reported
    /// success while discarding whatever the (also-fake) SYCL dispatch
    /// "computed". `execute_kernel` no longer reaches this (it now errors
    /// unconditionally, see above), but this is kept honest in its own
    /// right rather than left as a working-looking zero-fill.
    #[allow(dead_code)]
    fn create_output_tensors(&self, _inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        Err(compute_error(
            "oneapi_operation",
            "cannot construct real kernel output tensors: no SYCL runtime is linked to read \
             device output buffers back from",
        ))
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

    // `update_execution_metrics` (execution_time, metadata) used to live
    // here: unreachable (nothing on this backend can succeed - see `new`
    // above), and its body fabricated `utilization = 0.8` from nothing while
    // ignoring `metadata` entirely - a hardcoded number with no real signal
    // behind it and no caller to receive it. Unlike `create_output_tensors`
    // above, there was no honest replacement to give it (there is no real
    // GPU occupancy signal to derive `utilization` from without a runtime,
    // and `HardwareMetrics::utilization` is a plain `f64`, not an
    // `Option<f64>`, on a struct shared by every backend, so it cannot
    // honestly report "unknown" either), so it was deleted rather than kept
    // fabricating a number.
}

impl OneApiMemoryManager {
    fn new(optimization: MemoryOptimization) -> Self {
        Self {
            memory_pools: HashMap::new(),
            usm_allocations: HashMap::new(),
            optimization_strategy: optimization,
        }
    }

    /// Allocate unified shared memory.
    ///
    /// Always errors: no SYCL USM allocator is linked into this build (see
    /// the `extern "C"` block gated at the top of this module). Never
    /// dereferences `queue` since no real allocation is attempted.
    ///
    /// # Safety
    ///
    /// Kept `unsafe` to preserve the public API signature; there is
    /// currently no actual unsafe behavior since this never touches
    /// `queue`.
    pub unsafe fn allocate_usm(
        &mut self,
        _id: String,
        _size: usize,
        _usm_type: UsmType,
        _queue: *mut SyclQueue,
    ) -> HardwareResult<*mut u8> {
        Err(compute_error(
            "oneapi_operation",
            "Intel oneAPI USM allocation is not available: no SYCL runtime is linked",
        ))
    }

    /// Deallocate unified shared memory. Always errors (see `allocate_usm`).
    ///
    /// # Safety
    ///
    /// Kept `unsafe` to preserve the public API signature; there is
    /// currently no actual unsafe behavior since this never touches `queue`
    /// or any allocation (nothing can have been allocated via
    /// `allocate_usm`, which always errors).
    pub unsafe fn deallocate_usm(
        &mut self,
        _id: &str,
        _queue: *mut SyclQueue,
    ) -> HardwareResult<()> {
        Err(compute_error(
            "oneapi_operation",
            "Intel oneAPI USM deallocation is not available: no SYCL runtime is linked",
        ))
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
        // No real SYCL runtime is linked into this build (see the
        // `extern "C"` block above), and nothing in this module ever
        // constructs an `OneApiContext` with a non-null `queue` anymore
        // (`OneApiBackend::new` always errors before one would be built),
        // so there is nothing to destroy here.
    }
}

/// Utility functions for oneAPI integration
pub mod utils {
    use super::*;

    /// Check if Intel oneAPI is available.
    ///
    /// Always `false`: no SYCL runtime is linked into this build (see the
    /// `extern "C"` block gated at the top of this module). Previously this
    /// called `sycl_get_device_count`, an unresolved symbol.
    pub fn is_oneapi_available() -> bool {
        false
    }

    /// Get available oneAPI devices.
    ///
    /// Always empty: no SYCL runtime is linked into this build, so there is
    /// no real device to enumerate. Previously this called
    /// `sycl_get_device_count` (an unresolved symbol) and, for every unit
    /// reported, fabricated a device via `get_device_info` (vendor always
    /// `"Intel"`, `supports_dl_boost: true` unconditionally, etc.) - a
    /// phantom-device pattern this must not reproduce.
    pub fn get_available_devices() -> Vec<OneApiDevice> {
        Vec::new()
    }

    /// Generate optimized DPC++ kernel for matrix multiplication.
    ///
    /// `M`/`N`/`K` are runtime parameters of the emitted kernel, not
    /// compile-time constants baked into its body, so `m`/`n`/`k` do not
    /// change the generated arithmetic - but they document, in the source
    /// itself, which problem size this particular kernel text was generated
    /// for, rather than silently discarding the caller's stated shape.
    pub fn generate_gemm_kernel(m: usize, n: usize, k: usize) -> String {
        format!("\n// Generated for M={m}, N={n}, K={k}.")
            + r#"
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
    }

    /// Generate optimized DPC++ kernel for convolution.
    ///
    /// As with [`Self::generate_gemm_kernel`], the channel counts and kernel
    /// size are runtime parameters of the emitted kernel rather than
    /// compile-time constants, so `input_channels`/`output_channels`/
    /// `kernel_size` do not change the generated arithmetic - but they
    /// document, in the source itself, which configuration this kernel text
    /// was generated for.
    pub fn generate_conv2d_kernel(
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
    ) -> String {
        format!(
            "\n// Generated for input_channels={input_channels}, \
             output_channels={output_channels}, kernel_size={kernel_size}."
        ) + r#"
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
"#
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

    /// Regression test: `m`/`n`/`k` (and the conv2d channel/kernel-size
    /// parameters) used to be accepted and then completely ignored, so two
    /// calls with different problem sizes produced byte-identical source
    /// text. They must now show up in the generated source.
    #[test]
    fn test_kernel_generation_reflects_its_own_arguments() {
        let small = utils::generate_gemm_kernel(4, 8, 16);
        let large = utils::generate_gemm_kernel(400, 800, 1600);
        assert_ne!(
            small, large,
            "different M/N/K must produce different source text"
        );
        assert!(small.contains("M=4, N=8, K=16"));
        assert!(large.contains("M=400, N=800, K=1600"));

        let small_conv = utils::generate_conv2d_kernel(3, 16, 3);
        let large_conv = utils::generate_conv2d_kernel(64, 128, 5);
        assert_ne!(
            small_conv, large_conv,
            "different channel/kernel-size arguments must produce different source text"
        );
        assert!(small_conv.contains("input_channels=3, output_channels=16, kernel_size=3"));
        assert!(large_conv.contains("input_channels=64, output_channels=128, kernel_size=5"));
    }

    /// Regression test: `OneApiBackend::new()` used to "succeed" by calling
    /// `sycl_queue_create` (an unresolved extern symbol with no providing
    /// library) and building a context around whatever that returned. It
    /// must now honestly report the backend as unavailable.
    #[test]
    fn test_oneapi_backend_new_errors_no_real_runtime() {
        let result = OneApiBackend::new(OneApiConfig::default());
        assert!(
            result.is_err(),
            "must not fabricate a working oneAPI backend"
        );
    }

    /// Regression test: `is_oneapi_available` must not claim a SYCL runtime
    /// is present when none is linked into this build.
    #[test]
    fn test_is_oneapi_available_is_honest() {
        assert!(!utils::is_oneapi_available());
    }

    /// Regression test: device enumeration must return no phantom devices
    /// on a machine with no real oneAPI runtime.
    #[test]
    fn test_get_available_devices_reports_no_phantom_devices() {
        assert!(utils::get_available_devices().is_empty());
    }
}
