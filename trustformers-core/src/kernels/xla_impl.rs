// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! XLA (Accelerated Linear Algebra) backend implementation for TrustformeRS
//!
//! This module models the client/computation/buffer API surface (device
//! configuration, computation caching, buffer handles) that a real
//! integration with Google's XLA compiler would expose. It does **not**
//! link or execute a real XLA runtime: `--features xla` has zero
//! dependencies (`xla = []` in `Cargo.toml`), the `extern "C"` FFI
//! declarations further down are permanently disabled with `cfg(any())`,
//! and every public method that would need actual XLA compilation or
//! execution returns a structured [`HardwareResult`] error naming exactly
//! what is missing ("no XLA runtime is linked") instead of fabricating
//! output tensors, device counts, or timings. See the `HONESTY NOTE` above
//! the `extern "C"` block for the full rationale and what wiring a real
//! backend would require.

use crate::errors::compute_error;
use crate::hardware::{DataType, HardwareCapabilities, HardwareMetrics, HardwareResult};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// XLA computation backend
#[derive(Debug, Clone)]
pub struct XlaBackend {
    /// XLA client for compilation and execution
    client: Arc<XlaClient>,
    /// Device configuration
    device_config: XlaDeviceConfig,
    /// Compiled computations cache
    computation_cache: HashMap<String, XlaComputation>,
    /// Performance metrics
    metrics: HardwareMetrics,
}

/// XLA client for interfacing with XLA runtime
#[derive(Debug)]
pub struct XlaClient {
    /// XLA platform (CPU, GPU, TPU)
    #[allow(dead_code)]
    platform: XlaPlatform,
    /// Device ordinal
    #[allow(dead_code)]
    device_ordinal: i32,
    /// Client handle. Never non-null in practice: `XlaClient::new` always
    /// errors before constructing one (see the `extern "C"` honesty note
    /// above).
    #[allow(dead_code)]
    handle: *mut XlaClientHandle,
    /// Device memory allocator
    #[allow(dead_code)]
    allocator: XlaAllocator,
}

// SAFETY: XLA runtime handles are thread-safe internally
unsafe impl Send for XlaClient {}
unsafe impl Sync for XlaClient {}

/// XLA platform enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XlaPlatform {
    /// CPU execution
    CPU,
    /// CUDA GPU execution
    GPU,
    /// TPU execution
    TPU,
    /// Custom platform
    Custom(u32),
}

/// XLA device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XlaDeviceConfig {
    /// Platform type
    pub platform: XlaPlatform,
    /// Device ordinal
    pub device_ordinal: i32,
    /// Memory pool size in bytes
    pub memory_pool_size: Option<usize>,
    /// Enable auto-tuning
    pub enable_auto_tuning: bool,
    /// Optimization level (0-3)
    pub optimization_level: u32,
    /// Enable fusion optimizations
    pub enable_fusion: bool,
    /// Custom configuration options
    pub custom_options: HashMap<String, String>,
}

/// Compiled XLA computation
#[derive(Debug, Clone)]
pub struct XlaComputation {
    /// Computation name
    #[allow(dead_code)]
    name: String,
    /// Compiled executable
    executable: Arc<XlaExecutable>,
    /// Input shapes and types
    #[allow(dead_code)]
    input_spec: Vec<XlaShapeSpec>,
    /// Output shapes and types
    #[allow(dead_code)]
    output_spec: Vec<XlaShapeSpec>,
    /// Compilation metadata
    metadata: XlaCompilationMetadata,
}

/// XLA executable handle
#[derive(Debug)]
pub struct XlaExecutable {
    /// Executable handle. Never constructed with a real handle: nothing
    /// builds an `XlaExecutable` anymore (`XlaClient::compile` always
    /// errors first).
    #[allow(dead_code)]
    handle: *mut XlaExecutableHandle,
    /// Platform
    #[allow(dead_code)]
    platform: XlaPlatform,
    /// Device ordinal
    #[allow(dead_code)]
    device_ordinal: i32,
}

// SAFETY: XLA executable handles are thread-safe internally
unsafe impl Send for XlaExecutable {}
unsafe impl Sync for XlaExecutable {}

/// XLA shape specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct XlaShapeSpec {
    /// Element type
    pub element_type: DataType,
    /// Dimensions
    pub dimensions: Vec<i64>,
    /// Layout specification
    pub layout: Option<XlaLayout>,
}

/// XLA tensor layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XlaLayout {
    /// Minor-to-major dimension ordering
    pub minor_to_major: Vec<i32>,
    /// Tile configuration
    pub tiles: Vec<XlaTile>,
}

/// XLA tile specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XlaTile {
    /// Tile dimensions
    pub dimensions: Vec<i64>,
}

/// XLA compilation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XlaCompilationMetadata {
    /// Compilation time in milliseconds
    pub compilation_time_ms: f64,
    /// Number of operations
    pub operation_count: usize,
    /// Memory usage estimate in bytes
    pub memory_usage_bytes: usize,
    /// Flop count estimate
    pub flop_count: u64,
    /// Optimization passes applied
    pub optimization_passes: Vec<String>,
}

/// XLA memory allocator
#[derive(Debug)]
pub struct XlaAllocator {
    /// Platform
    #[allow(dead_code)]
    platform: XlaPlatform,
    /// Total memory size
    #[allow(dead_code)]
    total_memory: usize,
    /// Used memory
    #[allow(dead_code)]
    used_memory: usize,
    /// Memory fragmentation ratio
    #[allow(dead_code)]
    fragmentation: f64,
}

/// XLA buffer for tensor data
#[derive(Debug)]
pub struct XlaBuffer {
    /// Buffer handle. Never constructed with a real handle: nothing builds
    /// an `XlaBuffer` anymore (`XlaBackend::create_input_buffers` always
    /// errors first).
    #[allow(dead_code)]
    handle: *mut XlaBufferHandle,
    /// Shape specification
    #[allow(dead_code)]
    shape: XlaShapeSpec,
    /// Device ordinal
    #[allow(dead_code)]
    device_ordinal: i32,
    /// Size in bytes
    #[allow(dead_code)]
    size_bytes: usize,
}

// Foreign function interface declarations for XLA runtime.
//
// HONESTY NOTE: these symbols are declared but nothing in this workspace
// provides them - there is no `-sys` crate, no `#[link(name = "...")]`
// attribute, and `build.rs` links only `framework=Accelerate` on macOS.
// `--features xla` has zero dependencies (`xla = []` in Cargo.toml), so any
// real XLA runtime call here would be an unresolved symbol at final link
// time (independent of the fact that `XlaShapeSpec` containing a `Vec` is
// also not FFI-safe, noted below). This block (and every call site) is
// therefore gated behind `cfg(any())` - permanently disabled - so the crate
// never tries to link against a runtime that was never provided. Every
// public method that used to call into it now returns a structured error
// instead of silently returning fabricated data (zeroed/`[1,1]`-shaped
// output "tensors", invented device counts, ...). Wiring a real backend
// means adding a genuine `-sys` binding crate, a `#[link(...)]` target, a
// `build.rs` probe, FFI-safe shape types, and replacing the `cfg(any())`
// gate below with a real feature check.
// Note: XlaShapeSpec contains Vec which is not FFI-safe. These are placeholder
// declarations that would need proper C-compatible types in production use.
#[cfg(any())]
#[allow(improper_ctypes)]
extern "C" {
    fn xla_client_create(platform: i32, device_ordinal: i32) -> *mut XlaClientHandle;
    fn xla_client_destroy(client: *mut XlaClientHandle);
    fn xla_compile_computation(
        client: *mut XlaClientHandle,
        computation_text: *const i8,
        input_shapes: *const XlaShapeSpec,
        input_count: usize,
    ) -> *mut XlaExecutableHandle;
    fn xla_execute(
        executable: *mut XlaExecutableHandle,
        inputs: *const *mut XlaBufferHandle,
        input_count: usize,
        outputs: *mut *mut XlaBufferHandle,
        output_count: usize,
    ) -> i32;
    fn xla_buffer_create(
        client: *mut XlaClientHandle,
        data: *const f32,
        shape: *const XlaShapeSpec,
        device_ordinal: i32,
    ) -> *mut XlaBufferHandle;
    fn xla_buffer_destroy(buffer: *mut XlaBufferHandle);
    fn xla_buffer_to_host(buffer: *mut XlaBufferHandle, data: *mut f32, size: usize) -> i32;
    fn xla_get_platform_count() -> i32;
    fn xla_get_device_count(platform: i32) -> i32;
    fn xla_synchronize_device(device_ordinal: i32) -> i32;
}

// Opaque handle types for FFI
#[repr(C)]
pub struct XlaClientHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct XlaExecutableHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct XlaBufferHandle {
    _private: [u8; 0],
}

impl XlaBackend {
    /// Create a new XLA backend
    pub fn new(config: XlaDeviceConfig) -> HardwareResult<Self> {
        let client = Arc::new(XlaClient::new(config.platform, config.device_ordinal)?);

        let metrics = HardwareMetrics {
            ops_per_second: 0.0,
            memory_bandwidth: match config.platform {
                XlaPlatform::CPU => 100e9,       // 100 GB/s for CPU
                XlaPlatform::GPU => 1e12,        // 1 TB/s for GPU
                XlaPlatform::TPU => 1.2e12,      // 1.2 TB/s for TPU
                XlaPlatform::Custom(_) => 500e9, // 500 GB/s default
            },
            utilization: 0.0,
            power_consumption: 0.0,
            temperature: None,
            error_rate: 0.0,
            latency: 0.0,
            throughput: 0.0,
        };

        Ok(Self {
            client,
            device_config: config,
            computation_cache: HashMap::new(),
            metrics,
        })
    }

    /// Compile a tensor operation to XLA
    pub fn compile_operation(
        &mut self,
        operation_name: &str,
        hlo_text: &str,
        input_shapes: &[XlaShapeSpec],
    ) -> HardwareResult<String> {
        let computation_id = format!("{}_{}", operation_name, input_shapes.len());

        if self.computation_cache.contains_key(&computation_id) {
            return Ok(computation_id);
        }

        let start_time = std::time::Instant::now();

        let executable = self.client.compile(hlo_text, input_shapes)?;

        let compilation_time = start_time.elapsed().as_millis() as f64;

        let metadata = XlaCompilationMetadata {
            compilation_time_ms: compilation_time,
            operation_count: hlo_text.matches("f32[").count(),
            memory_usage_bytes: input_shapes.iter().map(|s| s.size_bytes()).sum(),
            flop_count: self.estimate_flops(hlo_text),
            optimization_passes: vec![
                "constant_folding".to_string(),
                "algebraic_simplifier".to_string(),
                "layout_assignment".to_string(),
                "buffer_assignment".to_string(),
            ],
        };

        let computation = XlaComputation {
            name: operation_name.to_string(),
            executable,
            input_spec: input_shapes.to_vec(),
            output_spec: self.infer_output_shapes(hlo_text, input_shapes)?,
            metadata,
        };

        self.computation_cache.insert(computation_id.clone(), computation);
        Ok(computation_id)
    }

    /// Execute a compiled computation
    pub fn execute_computation(
        &mut self,
        computation_id: &str,
        inputs: &[Tensor],
    ) -> HardwareResult<Vec<Tensor>> {
        let computation = self
            .computation_cache
            .get(computation_id)
            .ok_or_else(|| compute_error("execute_computation", "Computation not found"))?;

        let input_buffers = self.create_input_buffers(inputs)?;
        let output_buffers = self.client.execute(&computation.executable, &input_buffers)?;
        let outputs = self.buffers_to_tensors(output_buffers)?;

        // Update metrics
        let metadata = computation.metadata.clone();
        self.update_metrics(&metadata);

        Ok(outputs)
    }

    /// Get XLA backend capabilities
    pub fn get_capabilities(&self) -> HardwareCapabilities {
        let data_types = match self.device_config.platform {
            XlaPlatform::CPU => vec![
                DataType::F32,
                DataType::F64,
                DataType::I32,
                DataType::I64,
                DataType::Bool,
                DataType::Complex64,
                DataType::Complex128,
            ],
            XlaPlatform::GPU => vec![
                DataType::F32,
                DataType::F16,
                DataType::BF16,
                DataType::I32,
                DataType::I64,
                DataType::Bool,
                DataType::Complex64,
            ],
            XlaPlatform::TPU => vec![DataType::F32, DataType::BF16, DataType::I32, DataType::Bool],
            XlaPlatform::Custom(_) => vec![DataType::F32, DataType::I32],
        };

        HardwareCapabilities {
            data_types,
            max_dimensions: 8,
            memory_size: self.device_config.memory_pool_size,
            clock_frequency: None,
            compute_units: Some(match self.device_config.platform {
                XlaPlatform::CPU => 64,  // CPU cores
                XlaPlatform::GPU => 108, // GPU SMs
                XlaPlatform::TPU => 2,   // TPU cores
                XlaPlatform::Custom(_) => 32,
            }),
            operations: vec![
                "add".to_string(),
                "multiply".to_string(),
                "matmul".to_string(),
                "conv2d".to_string(),
                "reduce".to_string(),
                "transpose".to_string(),
                "reshape".to_string(),
                "slice".to_string(),
                "concatenate".to_string(),
                "broadcast".to_string(),
                "attention".to_string(),
            ],
            power_consumption: Some(match self.device_config.platform {
                XlaPlatform::CPU => 150.0, // 150W
                XlaPlatform::GPU => 300.0, // 300W
                XlaPlatform::TPU => 200.0, // 200W
                XlaPlatform::Custom(_) => 100.0,
            }),
            thermal_design_power: Some(match self.device_config.platform {
                XlaPlatform::CPU => 200.0, // 200W TDP
                XlaPlatform::GPU => 400.0, // 400W TDP
                XlaPlatform::TPU => 250.0, // 250W TDP
                XlaPlatform::Custom(_) => 150.0,
            }),
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &HardwareMetrics {
        &self.metrics
    }

    /// Optimize computation for target platform
    pub fn optimize_for_platform(&mut self, computation_id: &str) -> HardwareResult<()> {
        if let Some(computation) = self.computation_cache.get_mut(computation_id) {
            // Apply platform-specific optimizations
            match self.device_config.platform {
                XlaPlatform::CPU => {
                    // CPU-specific optimizations: vectorization, cache blocking
                    computation.metadata.optimization_passes.push("cpu_vectorization".to_string());
                    computation.metadata.optimization_passes.push("cache_blocking".to_string());
                },
                XlaPlatform::GPU => {
                    // GPU-specific optimizations: kernel fusion, memory coalescing
                    computation.metadata.optimization_passes.push("gpu_kernel_fusion".to_string());
                    computation.metadata.optimization_passes.push("memory_coalescing".to_string());
                },
                XlaPlatform::TPU => {
                    // TPU-specific optimizations: systolic array utilization
                    computation
                        .metadata
                        .optimization_passes
                        .push("tpu_systolic_optimization".to_string());
                    computation.metadata.optimization_passes.push("bfloat16_promotion".to_string());
                },
                XlaPlatform::Custom(_) => {
                    // Custom optimizations
                    computation
                        .metadata
                        .optimization_passes
                        .push("custom_optimization".to_string());
                },
            }
        }
        Ok(())
    }

    // Private helper methods.
    //
    // `create_input_buffers`/`buffers_to_tensors` used to call
    // `xla_buffer_create`/`xla_buffer_to_host` (unresolved symbols, see the
    // `extern "C"` block above) and are unreachable in practice now
    // (`XlaBackend::new` -> `XlaClient::new` always errors before any
    // `XlaBackend` exists to call them on), so they honestly error instead.
    fn create_input_buffers(&self, _inputs: &[Tensor]) -> HardwareResult<Vec<XlaBuffer>> {
        Err(compute_error(
            "xla_operation",
            "XLA buffer allocation is not available: no XLA runtime is linked",
        ))
    }

    fn buffers_to_tensors(&self, _buffers: Vec<XlaBuffer>) -> HardwareResult<Vec<Tensor>> {
        Err(compute_error(
            "xla_operation",
            "XLA buffer readback is not available: no XLA runtime is linked",
        ))
    }

    /// Real HLO output-shape inference (parsing the HLO text's `ROOT`
    /// instruction and computing its resulting shape) is not implemented;
    /// this used to silently return a fabricated `[1, 1]` shape regardless
    /// of the real output shape. Unreachable in practice now
    /// (`compile_operation` errors via `self.client.compile(...)?` before
    /// reaching this), so this honestly errors rather than guessing.
    fn infer_output_shapes(
        &self,
        _hlo_text: &str,
        _input_shapes: &[XlaShapeSpec],
    ) -> HardwareResult<Vec<XlaShapeSpec>> {
        Err(compute_error(
            "xla_operation",
            "XLA output shape inference is not implemented",
        ))
    }

    fn estimate_flops(&self, hlo_text: &str) -> u64 {
        // Simplified FLOP estimation based on HLO operations
        let matmul_count = hlo_text.matches("dot").count() as u64;
        let add_count = hlo_text.matches("add").count() as u64;
        let mul_count = hlo_text.matches("multiply").count() as u64;

        // Rough FLOP estimates
        matmul_count * 1000000 + add_count * 1000 + mul_count * 1000
    }

    fn update_metrics(&mut self, metadata: &XlaCompilationMetadata) {
        self.metrics.ops_per_second =
            metadata.flop_count as f64 / (metadata.compilation_time_ms / 1000.0);
        self.metrics.latency = metadata.compilation_time_ms;
        self.metrics.throughput = self.metrics.ops_per_second;
    }
}

impl XlaClient {
    /// Always errors: no XLA runtime is linked into this build (see the
    /// `extern "C"` block above, permanently gated with `cfg(any())`).
    /// Previously this called `xla_client_create`, an unresolved symbol,
    /// and on a null return (which is all it could ever be) still built an
    /// `XlaAllocator` advertising a fabricated per-platform memory size.
    fn new(_platform: XlaPlatform, _device_ordinal: i32) -> HardwareResult<Self> {
        Err(compute_error(
            "xla_operation",
            "XLA backend is not available: no XLA runtime is linked into this build (the \
             `xla` feature has no real backend binding yet)",
        ))
    }

    /// Always errors (see `XlaClient::new`): no XLA compiler is linked.
    /// Unreachable in practice since `Self` can never be constructed, kept
    /// honest in its own right rather than left calling an unresolved
    /// symbol.
    fn compile(
        &self,
        _hlo_text: &str,
        _input_shapes: &[XlaShapeSpec],
    ) -> HardwareResult<Arc<XlaExecutable>> {
        Err(compute_error(
            "xla_operation",
            "XLA compilation is not available: no XLA runtime is linked",
        ))
    }

    /// Always errors (see `XlaClient::new`). Previously this returned a
    /// buffer shaped `[1, 1]` regardless of the real output shape whenever
    /// the (unresolved) `xla_execute` call happened to leave a non-null
    /// handle behind.
    fn execute(
        &self,
        _executable: &XlaExecutable,
        _inputs: &[XlaBuffer],
    ) -> HardwareResult<Vec<XlaBuffer>> {
        Err(compute_error(
            "xla_operation",
            "XLA execution is not available: no XLA runtime is linked",
        ))
    }
}

impl XlaShapeSpec {
    /// Calculate size in bytes for this shape
    pub fn size_bytes(&self) -> usize {
        let element_size = match self.element_type {
            DataType::F32 | DataType::I32 => 4,
            DataType::F64 | DataType::I64 | DataType::Complex64 => 8,
            DataType::F16 | DataType::BF16 | DataType::I16 => 2,
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
            DataType::Complex128 => 16,
            _ => 4, // Default to 4 bytes
        };

        let element_count: usize = self.dimensions.iter().map(|&d| d as usize).product();
        element_count * element_size
    }
}

impl Default for XlaDeviceConfig {
    fn default() -> Self {
        Self {
            platform: XlaPlatform::CPU,
            device_ordinal: 0,
            memory_pool_size: None,
            enable_auto_tuning: true,
            optimization_level: 2,
            enable_fusion: true,
            custom_options: HashMap::new(),
        }
    }
}

impl Drop for XlaClient {
    fn drop(&mut self) {
        // No real XLA runtime is linked into this build, and `XlaClient`
        // can never be constructed with a non-null `handle` anymore
        // (`XlaClient::new` always errors first), so there is nothing to
        // destroy here.
    }
}

impl Drop for XlaBuffer {
    fn drop(&mut self) {
        // Likewise: `XlaBuffer` is never constructed with a real handle
        // (see `XlaBackend::create_input_buffers`), so nothing to destroy.
    }
}

/// Utility functions for XLA integration
pub mod utils {
    use super::*;

    /// Check if XLA is available on the system.
    ///
    /// Always `false`: no XLA runtime is linked into this build (see the
    /// `extern "C"` block gated at the top of this module). Previously
    /// this called `xla_get_platform_count`, an unresolved symbol.
    pub fn is_xla_available() -> bool {
        false
    }

    /// Get available XLA platforms.
    ///
    /// Always empty: no XLA runtime is linked into this build, so there is
    /// no real platform to enumerate. Previously this called
    /// `xla_get_platform_count` (an unresolved symbol) and fabricated a
    /// `CPU`/`GPU`/`TPU`/`Custom` platform for every unit it claimed to
    /// find - a phantom-device pattern this must not reproduce.
    pub fn get_available_platforms() -> Vec<XlaPlatform> {
        Vec::new()
    }

    /// Get device count for a platform.
    ///
    /// Always `0`: no XLA runtime is linked into this build. Previously
    /// called `xla_get_device_count`, an unresolved symbol.
    pub fn get_device_count(_platform: XlaPlatform) -> i32 {
        0
    }

    /// Synchronize device execution.
    ///
    /// Always errors: no XLA runtime is linked into this build. Previously
    /// called `xla_synchronize_device`, an unresolved symbol.
    pub fn synchronize_device(_device_ordinal: i32) -> HardwareResult<()> {
        Err(compute_error(
            "xla_operation",
            "XLA device synchronization is not available: no XLA runtime is linked",
        ))
    }

    /// Create optimized HLO for common operations
    pub fn create_matmul_hlo(lhs_shape: &[i64], rhs_shape: &[i64]) -> String {
        format!(
            r#"
HloModule matmul_module

ENTRY main {{
  lhs = f32[{}] parameter(0)
  rhs = f32[{}] parameter(1)
  ROOT result = f32[{},{}] dot(lhs, rhs), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
}}
"#,
            lhs_shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            rhs_shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            lhs_shape[0],
            rhs_shape[1]
        )
    }

    /// Create HLO for convolution operation
    pub fn create_conv2d_hlo(
        input_shape: &[i64],
        kernel_shape: &[i64],
        strides: &[i64],
        padding: &[i64],
    ) -> String {
        format!(
            r#"
HloModule conv2d_module

ENTRY main {{
  input = f32[{}] parameter(0)
  kernel = f32[{}] parameter(1)
  ROOT result = f32[{},{},{},{}] convolution(input, kernel),
    window={{size={}x{} stride={}x{} pad={}_{}_{}_{}}},
    dim_labels=b01f_01io->b01f
}}
"#,
            input_shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            kernel_shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            input_shape[0], // batch
            (input_shape[1] + 2 * padding[0] - kernel_shape[0]) / strides[0] + 1, // height
            (input_shape[2] + 2 * padding[1] - kernel_shape[1]) / strides[1] + 1, // width
            kernel_shape[3], // output channels
            kernel_shape[0],
            kernel_shape[1], // kernel size
            strides[0],
            strides[1], // strides
            padding[0],
            padding[0],
            padding[1],
            padding[1] // padding
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xla_platform_serialization() {
        let platform = XlaPlatform::TPU;
        let serialized = serde_json::to_string(&platform).expect("JSON serialization failed");
        let deserialized: XlaPlatform =
            serde_json::from_str(&serialized).expect("JSON deserialization failed");
        assert_eq!(platform, deserialized);
    }

    #[test]
    fn test_xla_device_config_default() {
        let config = XlaDeviceConfig::default();
        assert_eq!(config.platform, XlaPlatform::CPU);
        assert_eq!(config.device_ordinal, 0);
        assert!(config.enable_auto_tuning);
        assert_eq!(config.optimization_level, 2);
    }

    #[test]
    fn test_xla_shape_spec_size_calculation() {
        let shape = XlaShapeSpec {
            element_type: DataType::F32,
            dimensions: vec![2, 3, 4],
            layout: None,
        };
        assert_eq!(shape.size_bytes(), 2 * 3 * 4 * 4); // 96 bytes
    }

    #[test]
    fn test_xla_utils_hlo_generation() {
        let hlo = utils::create_matmul_hlo(&[2, 3], &[3, 4]);
        assert!(hlo.contains("dot"));
        assert!(hlo.contains("f32[2,4]"));
    }

    #[test]
    fn test_xla_conv2d_hlo_generation() {
        let hlo = utils::create_conv2d_hlo(&[1, 28, 28, 3], &[3, 3, 3, 32], &[1, 1], &[1, 1]);
        assert!(hlo.contains("convolution"));
        assert!(hlo.contains("window"));
        assert!(hlo.contains("dim_labels"));
    }

    /// Regression test: `XlaBackend::new()` (via `XlaClient::new`) used to
    /// "succeed" by calling `xla_client_create` (an unresolved extern
    /// symbol with no providing library). It must now honestly report the
    /// backend as unavailable.
    #[test]
    fn test_xla_backend_new_errors_no_real_runtime() {
        let result = XlaBackend::new(XlaDeviceConfig::default());
        assert!(result.is_err(), "must not fabricate a working XLA backend");
    }

    /// Regression test: `is_xla_available` must not claim an XLA runtime is
    /// present when none is linked into this build.
    #[test]
    fn test_is_xla_available_is_honest() {
        assert!(!utils::is_xla_available());
    }

    /// Regression test: platform/device enumeration must report no phantom
    /// platforms or devices on a machine with no real XLA runtime.
    #[test]
    fn test_xla_enumeration_reports_no_phantom_platforms() {
        assert!(utils::get_available_platforms().is_empty());
        assert_eq!(utils::get_device_count(XlaPlatform::CPU), 0);
        assert_eq!(utils::get_device_count(XlaPlatform::GPU), 0);
    }
}
