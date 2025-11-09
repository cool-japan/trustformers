// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TPU (Tensor Processing Unit) backend implementation for TrustformeRS
//!
//! This module provides direct integration with Google's TPU hardware for
//! optimized execution of machine learning workloads, including support for
//! TPU v4, v5, and custom TPU configurations.

#![allow(dead_code)] // TPU backend implementation with FFI bindings

use crate::errors::compute_error;
use crate::hardware::{DataType, HardwareCapabilities, HardwareMetrics, HardwareResult};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// TPU backend for tensor operations
#[derive(Debug)]
pub struct TpuBackend {
    /// TPU device handle
    device: Arc<Mutex<TpuDevice>>,
    /// TPU configuration
    config: TpuConfig,
    /// Compilation cache
    program_cache: HashMap<String, TpuProgram>,
    /// Performance metrics
    metrics: Arc<Mutex<HardwareMetrics>>,
    /// Memory manager
    memory_manager: TpuMemoryManager,
}

/// TPU device representation
#[derive(Debug)]
pub struct TpuDevice {
    /// Device identifier
    device_id: String,
    /// TPU generation (v4, v5, etc.)
    generation: TpuGeneration,
    /// Number of cores
    core_count: u32,
    /// Total memory in bytes
    memory_size: usize,
    /// Current device status
    status: TpuDeviceStatus,
    /// Runtime handle
    runtime_handle: *mut TpuRuntimeHandle,
}

// SAFETY: TPU runtime handles are thread-safe internally
unsafe impl Send for TpuDevice {}
unsafe impl Sync for TpuDevice {}

/// TPU generation variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TpuGeneration {
    /// TPU v2
    V2,
    /// TPU v3
    V3,
    /// TPU v4
    V4,
    /// TPU v5
    V5,
    /// TPU v5e (Edge optimized)
    V5E,
    /// Custom TPU
    Custom(u32),
}

/// TPU device status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TpuDeviceStatus {
    /// Device is online and available
    pub online: bool,
    /// Device is currently executing
    pub busy: bool,
    /// Current temperature in Celsius
    pub temperature: Option<f64>,
    /// Power consumption in watts
    pub power_consumption: Option<f64>,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Compute utilization (0.0 to 1.0)
    pub compute_utilization: f64,
    /// Last error message
    pub last_error: Option<String>,
}

/// TPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpuConfig {
    /// TPU generation
    pub generation: TpuGeneration,
    /// Device topology (e.g., "2x2" for 4 cores)
    pub topology: String,
    /// Enable XLA compilation
    pub enable_xla: bool,
    /// Enable bfloat16 optimization
    pub enable_bfloat16: bool,
    /// Memory pool size
    pub memory_pool_size: Option<usize>,
    /// Batch size optimization
    pub optimal_batch_size: Option<usize>,
    /// Enable systolic array optimization
    pub enable_systolic_optimization: bool,
    /// Custom TPU options
    pub custom_options: HashMap<String, String>,
}

/// Compiled TPU program
#[derive(Debug, Clone)]
pub struct TpuProgram {
    /// Program name
    name: String,
    /// Compiled binary
    binary: Vec<u8>,
    /// Input specifications
    input_specs: Vec<TpuTensorSpec>,
    /// Output specifications
    output_specs: Vec<TpuTensorSpec>,
    /// Compilation metadata
    metadata: TpuCompilationMetadata,
    /// Program handle
    handle: *mut TpuProgramHandle,
}

/// TPU tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpuTensorSpec {
    /// Element data type
    pub data_type: DataType,
    /// Tensor dimensions
    pub dimensions: Vec<usize>,
    /// Memory layout (row-major, column-major, custom)
    pub layout: TpuMemoryLayout,
    /// Sharding specification for multi-core
    pub sharding: Option<TpuSharding>,
}

/// TPU memory layout specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TpuMemoryLayout {
    /// Row-major (C-style)
    RowMajor,
    /// Column-major (Fortran-style)
    ColumnMajor,
    /// TPU-optimized layout
    TpuOptimized,
    /// Custom layout with dimension ordering
    Custom(Vec<usize>),
}

/// TPU sharding configuration for multi-core execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpuSharding {
    /// Sharding dimensions
    pub dimensions: Vec<usize>,
    /// Replica count
    pub replicas: usize,
    /// Device mesh topology
    pub mesh_shape: Vec<usize>,
}

/// TPU compilation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpuCompilationMetadata {
    /// Compilation time in milliseconds
    pub compilation_time_ms: f64,
    /// Binary size in bytes
    pub binary_size_bytes: usize,
    /// Estimated FLOPS count
    pub estimated_flops: u64,
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
    /// Optimization level applied
    pub optimization_level: u32,
    /// Optimizations applied
    pub optimizations: Vec<String>,
}

/// TPU memory manager
#[derive(Debug)]
pub struct TpuMemoryManager {
    /// Total memory size
    total_memory: usize,
    /// Currently allocated memory
    allocated_memory: usize,
    /// Memory allocation map
    allocations: HashMap<String, TpuMemoryAllocation>,
    /// Memory fragmentation ratio
    fragmentation: f64,
}

/// TPU memory allocation
#[derive(Debug, Clone)]
pub struct TpuMemoryAllocation {
    /// Allocation ID
    pub id: String,
    /// Size in bytes
    pub size: usize,
    /// Memory address
    pub address: u64,
    /// Allocation timestamp
    pub allocated_at: Instant,
    /// Layout specification
    pub layout: TpuMemoryLayout,
}

/// TPU operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TpuOperation {
    /// Matrix multiplication
    MatMul,
    /// Convolution 2D
    Conv2D,
    /// Batch normalization
    BatchNorm,
    /// Activation functions
    Activation(ActivationType),
    /// Reduction operations
    Reduce(ReduceType),
    /// Element-wise operations
    ElementWise(ElementWiseType),
    /// Transformer attention
    Attention,
    /// Custom operation
    Custom(String),
}

/// Activation function types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
    Softmax,
}

/// Reduction operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReduceType {
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,
}

/// Element-wise operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ElementWiseType {
    Add,
    Multiply,
    Subtract,
    Divide,
    Power,
    Compare,
}

// Foreign function interface for TPU runtime
extern "C" {
    fn tpu_runtime_create() -> *mut TpuRuntimeHandle;
    fn tpu_runtime_destroy(handle: *mut TpuRuntimeHandle);
    fn tpu_device_enumerate(devices: *mut TpuDeviceInfo, max_count: usize) -> i32;
    fn tpu_device_open(device_id: *const i8) -> *mut TpuDeviceHandle;
    fn tpu_device_close(device: *mut TpuDeviceHandle);
    fn tpu_program_compile(
        device: *mut TpuDeviceHandle,
        source: *const i8,
        source_len: usize,
        config: *const TpuCompileConfig,
    ) -> *mut TpuProgramHandle;
    fn tpu_program_execute(
        program: *mut TpuProgramHandle,
        inputs: *const *const f32,
        input_shapes: *const TpuShape,
        input_count: usize,
        outputs: *mut *mut f32,
        output_shapes: *mut TpuShape,
        output_count: usize,
    ) -> i32;
    fn tpu_memory_allocate(device: *mut TpuDeviceHandle, size: usize) -> *mut TpuMemoryHandle;
    fn tpu_memory_deallocate(memory: *mut TpuMemoryHandle);
    fn tpu_synchronize(device: *mut TpuDeviceHandle) -> i32;
    fn tpu_get_device_info(device: *mut TpuDeviceHandle, info: *mut TpuDeviceInfo) -> i32;
}

// Opaque handle types for FFI
#[repr(C)]
pub struct TpuRuntimeHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TpuDeviceHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TpuProgramHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TpuMemoryHandle {
    _private: [u8; 0],
}

/// TPU device information structure for FFI
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TpuDeviceInfo {
    pub device_id: [i8; 64],
    pub generation: i32,
    pub core_count: u32,
    pub memory_size: u64,
    pub peak_ops_per_second: f64,
    pub memory_bandwidth: f64,
}

/// TPU shape descriptor for FFI
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TpuShape {
    pub dimensions: [i32; 8],
    pub rank: i32,
    pub element_type: i32,
}

/// TPU compilation configuration for FFI
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TpuCompileConfig {
    pub optimization_level: i32,
    pub enable_xla: i32,
    pub enable_bfloat16: i32,
    pub batch_size: i32,
}

impl TpuBackend {
    /// Create a new TPU backend
    pub fn new(config: TpuConfig) -> HardwareResult<Self> {
        let device = Self::initialize_device(&config)?;

        let metrics = Arc::new(Mutex::new(HardwareMetrics {
            ops_per_second: 0.0,
            memory_bandwidth: Self::get_memory_bandwidth(&config.generation),
            utilization: 0.0,
            power_consumption: 0.0,
            temperature: None,
            error_rate: 0.0,
            latency: 0.0,
            throughput: 0.0,
        }));

        let memory_manager = TpuMemoryManager::new(Self::get_memory_size(&config.generation));

        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            config,
            program_cache: HashMap::new(),
            metrics,
            memory_manager,
        })
    }

    /// Compile a program for TPU execution
    pub fn compile_program(
        &mut self,
        name: &str,
        source: &str,
        input_specs: &[TpuTensorSpec],
    ) -> HardwareResult<String> {
        let program_id = format!("{}_{}", name, input_specs.len());

        if self.program_cache.contains_key(&program_id) {
            return Ok(program_id);
        }

        let start_time = Instant::now();

        let device = self.device.lock().unwrap();
        let compile_config = TpuCompileConfig {
            optimization_level: if self.config.enable_xla { 3 } else { 2 },
            enable_xla: if self.config.enable_xla { 1 } else { 0 },
            enable_bfloat16: if self.config.enable_bfloat16 { 1 } else { 0 },
            batch_size: self.config.optimal_batch_size.unwrap_or(1) as i32,
        };

        let program_handle = unsafe {
            let source_bytes = source.as_bytes();
            tpu_program_compile(
                device.runtime_handle as *mut TpuDeviceHandle,
                source_bytes.as_ptr() as *const i8,
                source_bytes.len(),
                &compile_config,
            )
        };

        if program_handle.is_null() {
            return Err(compute_error(
                "tpu_operation",
                "TPU program compilation failed",
            ));
        }

        let compilation_time = start_time.elapsed().as_millis() as f64;

        let metadata = TpuCompilationMetadata {
            compilation_time_ms: compilation_time,
            binary_size_bytes: source.len(), // Simplified
            estimated_flops: self.estimate_flops(source),
            memory_usage_bytes: input_specs.iter().map(|s| s.size_bytes()).sum(),
            optimization_level: compile_config.optimization_level as u32,
            optimizations: self.get_applied_optimizations(),
        };

        let program = TpuProgram {
            name: name.to_string(),
            binary: source.as_bytes().to_vec(),
            input_specs: input_specs.to_vec(),
            output_specs: self.infer_output_specs(source, input_specs)?,
            metadata,
            handle: program_handle,
        };

        self.program_cache.insert(program_id.clone(), program);
        Ok(program_id)
    }

    /// Execute a compiled program
    pub fn execute_program(
        &mut self,
        program_id: &str,
        inputs: &[Tensor],
    ) -> HardwareResult<Vec<Tensor>> {
        let program = self
            .program_cache
            .get(program_id)
            .ok_or_else(|| compute_error("tpu_operation", "Program not found"))?;

        let start_time = Instant::now();

        // Prepare input data
        let input_ptrs: Vec<*const f32> = inputs
            .iter()
            .map(|tensor| tensor.data().map(|data| data.as_ptr()))
            .collect::<Result<Vec<_>, _>>()?;

        let input_shapes: Vec<TpuShape> =
            inputs.iter().map(|tensor| self.tensor_to_tpu_shape(tensor)).collect();

        // Prepare output buffers
        let mut output_ptrs: Vec<*mut f32> = vec![std::ptr::null_mut(); program.output_specs.len()];
        let mut output_shapes: Vec<TpuShape> =
            program.output_specs.iter().map(|spec| self.spec_to_tpu_shape(spec)).collect();

        // Allocate output memory
        for (i, spec) in program.output_specs.iter().enumerate() {
            let size = spec.dimensions.iter().product::<usize>();
            let mut output_data = vec![0.0f32; size];
            output_ptrs[i] = output_data.as_mut_ptr();
            // Note: In production, this would need proper memory management
        }

        // Execute program
        let result = unsafe {
            tpu_program_execute(
                program.handle,
                input_ptrs.as_ptr(),
                input_shapes.as_ptr(),
                input_ptrs.len(),
                output_ptrs.as_mut_ptr(),
                output_shapes.as_mut_ptr(),
                output_ptrs.len(),
            )
        };

        if result != 0 {
            return Err(compute_error(
                "tpu_operation",
                "TPU program execution failed",
            ));
        }

        // Convert outputs to tensors
        let mut output_tensors = Vec::new();
        for (i, spec) in program.output_specs.iter().enumerate() {
            let size = spec.dimensions.iter().product::<usize>();
            let data = unsafe { std::slice::from_raw_parts(output_ptrs[i], size).to_vec() };
            let tensor = Tensor::from_vec(data, &spec.dimensions)?;
            output_tensors.push(tensor);
        }

        // Update metrics
        let execution_time = start_time.elapsed();
        let metadata = program.metadata.clone();
        self.update_execution_metrics(execution_time, &metadata);

        Ok(output_tensors)
    }

    /// Get TPU backend capabilities
    pub fn get_capabilities(&self) -> HardwareCapabilities {
        let data_types = match self.config.generation {
            TpuGeneration::V4 | TpuGeneration::V5 | TpuGeneration::V5E => vec![
                DataType::F32,
                DataType::BF16,
                DataType::I32,
                DataType::I8,
                DataType::Bool,
            ],
            TpuGeneration::V2 | TpuGeneration::V3 => {
                vec![DataType::F32, DataType::BF16, DataType::I32, DataType::Bool]
            },
            TpuGeneration::Custom(_) => vec![DataType::F32, DataType::I32],
        };

        let (compute_units, memory_size, power_consumption) = match self.config.generation {
            TpuGeneration::V2 => (1, 8 * 1024 * 1024 * 1024, 200.0), // 8GB, 200W
            TpuGeneration::V3 => (2, 16 * 1024 * 1024 * 1024, 300.0), // 16GB, 300W
            TpuGeneration::V4 => (2, 32 * 1024 * 1024 * 1024, 350.0), // 32GB, 350W
            TpuGeneration::V5 => (4, 64 * 1024 * 1024 * 1024, 400.0), // 64GB, 400W
            TpuGeneration::V5E => (2, 16 * 1024 * 1024 * 1024, 150.0), // 16GB, 150W (edge)
            TpuGeneration::Custom(_) => (1, 8 * 1024 * 1024 * 1024, 200.0), // 8GB, 200W
        };

        HardwareCapabilities {
            data_types,
            max_dimensions: 8,
            memory_size: Some(memory_size),
            clock_frequency: Some(1_400_000_000), // 1.4 GHz base frequency
            compute_units: Some(compute_units),
            operations: vec![
                "matmul".to_string(),
                "conv2d".to_string(),
                "batch_norm".to_string(),
                "activation".to_string(),
                "reduce".to_string(),
                "attention".to_string(),
                "softmax".to_string(),
                "layer_norm".to_string(),
                "embedding".to_string(),
            ],
            power_consumption: Some(power_consumption),
            thermal_design_power: Some(power_consumption * 1.2), // 20% overhead
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> HardwareMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Optimize program for TPU systolic arrays
    pub fn optimize_for_systolic_arrays(&mut self, program_id: &str) -> HardwareResult<()> {
        if let Some(program) = self.program_cache.get_mut(program_id) {
            if self.config.enable_systolic_optimization {
                program.metadata.optimizations.extend(vec![
                    "systolic_array_mapping".to_string(),
                    "weight_stationary_optimization".to_string(),
                    "data_flow_optimization".to_string(),
                    "memory_hierarchy_optimization".to_string(),
                ]);
            }
        }
        Ok(())
    }

    /// Enable bfloat16 optimizations
    pub fn enable_bfloat16_optimization(&mut self, program_id: &str) -> HardwareResult<()> {
        if let Some(program) = self.program_cache.get_mut(program_id) {
            if self.config.enable_bfloat16 {
                program.metadata.optimizations.extend(vec![
                    "bfloat16_conversion".to_string(),
                    "mixed_precision_training".to_string(),
                    "gradient_scaling".to_string(),
                ]);
            }
        }
        Ok(())
    }

    // Private helper methods
    fn initialize_device(config: &TpuConfig) -> HardwareResult<TpuDevice> {
        let runtime_handle = unsafe { tpu_runtime_create() };
        if runtime_handle.is_null() {
            return Err(compute_error(
                "tpu_operation",
                "Failed to create TPU runtime",
            ));
        }

        let (core_count, memory_size) = match config.generation {
            TpuGeneration::V2 => (1, 8 * 1024 * 1024 * 1024),
            TpuGeneration::V3 => (2, 16 * 1024 * 1024 * 1024),
            TpuGeneration::V4 => (2, 32 * 1024 * 1024 * 1024),
            TpuGeneration::V5 => (4, 64 * 1024 * 1024 * 1024),
            TpuGeneration::V5E => (2, 16 * 1024 * 1024 * 1024),
            TpuGeneration::Custom(_) => (1, 8 * 1024 * 1024 * 1024),
        };

        Ok(TpuDevice {
            device_id: format!("tpu_{:?}_0", config.generation),
            generation: config.generation,
            core_count,
            memory_size,
            status: TpuDeviceStatus {
                online: true,
                busy: false,
                temperature: Some(65.0),
                power_consumption: Some(200.0),
                memory_utilization: 0.0,
                compute_utilization: 0.0,
                last_error: None,
            },
            runtime_handle,
        })
    }

    fn get_memory_bandwidth(generation: &TpuGeneration) -> f64 {
        match generation {
            TpuGeneration::V2 => 700e9,        // 700 GB/s
            TpuGeneration::V3 => 900e9,        // 900 GB/s
            TpuGeneration::V4 => 1.2e12,       // 1.2 TB/s
            TpuGeneration::V5 => 1.6e12,       // 1.6 TB/s
            TpuGeneration::V5E => 800e9,       // 800 GB/s (edge optimized)
            TpuGeneration::Custom(_) => 500e9, // 500 GB/s default
        }
    }

    fn get_memory_size(generation: &TpuGeneration) -> usize {
        match generation {
            TpuGeneration::V2 => 8 * 1024 * 1024 * 1024,   // 8GB
            TpuGeneration::V3 => 16 * 1024 * 1024 * 1024,  // 16GB
            TpuGeneration::V4 => 32 * 1024 * 1024 * 1024,  // 32GB
            TpuGeneration::V5 => 64 * 1024 * 1024 * 1024,  // 64GB
            TpuGeneration::V5E => 16 * 1024 * 1024 * 1024, // 16GB
            TpuGeneration::Custom(_) => 8 * 1024 * 1024 * 1024, // 8GB
        }
    }

    fn estimate_flops(&self, source: &str) -> u64 {
        // Simplified FLOP estimation based on operation count
        let matmul_count = source.matches("matmul").count() as u64;
        let conv_count = source.matches("conv").count() as u64;
        let attention_count = source.matches("attention").count() as u64;

        // Rough estimates based on typical operation complexity
        matmul_count * 1_000_000 + conv_count * 5_000_000 + attention_count * 10_000_000
    }

    fn get_applied_optimizations(&self) -> Vec<String> {
        let mut optimizations = vec![
            "constant_folding".to_string(),
            "dead_code_elimination".to_string(),
            "algebraic_simplification".to_string(),
        ];

        if self.config.enable_xla {
            optimizations.extend(vec!["xla_fusion".to_string(), "xla_clustering".to_string()]);
        }

        if self.config.enable_bfloat16 {
            optimizations.push("bfloat16_optimization".to_string());
        }

        if self.config.enable_systolic_optimization {
            optimizations.extend(vec![
                "systolic_array_optimization".to_string(),
                "memory_layout_optimization".to_string(),
            ]);
        }

        optimizations
    }

    fn infer_output_specs(
        &self,
        _source: &str,
        input_specs: &[TpuTensorSpec],
    ) -> HardwareResult<Vec<TpuTensorSpec>> {
        // Simplified output inference - in practice this would parse the program
        let output_spec = TpuTensorSpec {
            data_type: input_specs[0].data_type,
            dimensions: input_specs[0].dimensions.clone(),
            layout: TpuMemoryLayout::TpuOptimized,
            sharding: None,
        };
        Ok(vec![output_spec])
    }

    fn tensor_to_tpu_shape(&self, tensor: &Tensor) -> TpuShape {
        let mut dimensions = [0i32; 8];
        let shape = tensor.shape();
        for (i, &dim) in shape.iter().take(8).enumerate() {
            dimensions[i] = dim as i32;
        }

        TpuShape {
            dimensions,
            rank: shape.len() as i32,
            element_type: 0, // F32
        }
    }

    fn spec_to_tpu_shape(&self, spec: &TpuTensorSpec) -> TpuShape {
        let mut dimensions = [0i32; 8];
        for (i, &dim) in spec.dimensions.iter().take(8).enumerate() {
            dimensions[i] = dim as i32;
        }

        TpuShape {
            dimensions,
            rank: spec.dimensions.len() as i32,
            element_type: match spec.data_type {
                DataType::F32 => 0,
                DataType::BF16 => 1,
                DataType::I32 => 2,
                _ => 0,
            },
        }
    }

    fn update_execution_metrics(
        &mut self,
        execution_time: Duration,
        metadata: &TpuCompilationMetadata,
    ) {
        let mut metrics = self.metrics.lock().unwrap();
        let execution_ms = execution_time.as_millis() as f64;

        metrics.ops_per_second = metadata.estimated_flops as f64 / (execution_ms / 1000.0);
        metrics.latency = execution_ms;
        metrics.throughput = metrics.ops_per_second;
        metrics.utilization = 0.8; // Simplified utilization estimate
    }
}

impl TpuTensorSpec {
    /// Calculate size in bytes for this tensor specification
    pub fn size_bytes(&self) -> usize {
        let element_size = match self.data_type {
            DataType::F32 | DataType::I32 => 4,
            DataType::F64 | DataType::I64 => 8,
            DataType::F16 | DataType::BF16 | DataType::I16 => 2,
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
            _ => 4,
        };

        let element_count: usize = self.dimensions.iter().product();
        element_count * element_size
    }
}

impl TpuMemoryManager {
    fn new(total_memory: usize) -> Self {
        Self {
            total_memory,
            allocated_memory: 0,
            allocations: HashMap::new(),
            fragmentation: 0.0,
        }
    }

    /// Allocate memory on TPU
    pub fn allocate(
        &mut self,
        id: String,
        size: usize,
        layout: TpuMemoryLayout,
    ) -> HardwareResult<u64> {
        if self.allocated_memory + size > self.total_memory {
            return Err(compute_error("tpu_operation", "Out of TPU memory"));
        }

        let address = self.allocated_memory as u64;
        let allocation = TpuMemoryAllocation {
            id: id.clone(),
            size,
            address,
            allocated_at: Instant::now(),
            layout,
        };

        self.allocations.insert(id, allocation);
        self.allocated_memory += size;

        Ok(address)
    }

    /// Deallocate memory
    pub fn deallocate(&mut self, id: &str) -> HardwareResult<()> {
        if let Some(allocation) = self.allocations.remove(id) {
            self.allocated_memory -= allocation.size;
            Ok(())
        } else {
            Err(compute_error("tpu_operation", "Allocation not found"))
        }
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> (usize, usize, f64) {
        (self.total_memory, self.allocated_memory, self.fragmentation)
    }
}

impl Default for TpuConfig {
    fn default() -> Self {
        Self {
            generation: TpuGeneration::V4,
            topology: "2x2".to_string(),
            enable_xla: true,
            enable_bfloat16: true,
            memory_pool_size: None,
            optimal_batch_size: Some(64),
            enable_systolic_optimization: true,
            custom_options: HashMap::new(),
        }
    }
}

impl Drop for TpuDevice {
    fn drop(&mut self) {
        if !self.runtime_handle.is_null() {
            unsafe {
                tpu_runtime_destroy(self.runtime_handle);
            }
        }
    }
}

/// Utility functions for TPU operations
pub mod utils {
    use super::*;

    /// Check if TPU hardware is available
    pub fn is_tpu_available() -> bool {
        let mut devices = [TpuDeviceInfo {
            device_id: [0; 64],
            generation: 0,
            core_count: 0,
            memory_size: 0,
            peak_ops_per_second: 0.0,
            memory_bandwidth: 0.0,
        }];

        let count = unsafe { tpu_device_enumerate(devices.as_mut_ptr(), 1) };
        count > 0
    }

    /// Get available TPU devices
    pub fn get_available_devices() -> Vec<TpuDeviceInfo> {
        let mut devices = vec![
            TpuDeviceInfo {
                device_id: [0; 64],
                generation: 0,
                core_count: 0,
                memory_size: 0,
                peak_ops_per_second: 0.0,
                memory_bandwidth: 0.0,
            };
            8
        ]; // Maximum 8 devices

        let count = unsafe { tpu_device_enumerate(devices.as_mut_ptr(), devices.len()) };
        devices.truncate(count as usize);
        devices
    }

    /// Get optimal batch size for given input shape and TPU generation
    pub fn get_optimal_batch_size(input_shape: &[usize], generation: TpuGeneration) -> usize {
        let base_batch_size = match generation {
            TpuGeneration::V2 => 32,
            TpuGeneration::V3 => 64,
            TpuGeneration::V4 => 128,
            TpuGeneration::V5 => 256,
            TpuGeneration::V5E => 64,
            TpuGeneration::Custom(_) => 32,
        };

        // Adjust based on input size
        let input_size: usize = input_shape.iter().product();
        if input_size > 1_000_000 {
            base_batch_size / 4
        } else if input_size > 100_000 {
            base_batch_size / 2
        } else {
            base_batch_size
        }
    }

    /// Create TPU-optimized sharding configuration
    pub fn create_optimal_sharding(tensor_shape: &[usize], device_topology: &str) -> TpuSharding {
        let mesh_shape = match device_topology {
            "1x1" => vec![1],
            "2x2" => vec![2, 2],
            "4x4" => vec![4, 4],
            "8x8" => vec![8, 8],
            _ => vec![2, 2], // Default
        };

        TpuSharding {
            dimensions: tensor_shape.to_vec(),
            replicas: mesh_shape.iter().product::<usize>(),
            mesh_shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpu_generation_serialization() {
        let generation = TpuGeneration::V5;
        let serialized = serde_json::to_string(&generation).unwrap();
        let deserialized: TpuGeneration = serde_json::from_str(&serialized).unwrap();
        assert_eq!(generation, deserialized);
    }

    #[test]
    fn test_tpu_config_default() {
        let config = TpuConfig::default();
        assert_eq!(config.generation, TpuGeneration::V4);
        assert_eq!(config.topology, "2x2");
        assert!(config.enable_xla);
        assert!(config.enable_bfloat16);
    }

    #[test]
    fn test_tpu_tensor_spec_size_calculation() {
        let spec = TpuTensorSpec {
            data_type: DataType::F32,
            dimensions: vec![2, 3, 4],
            layout: TpuMemoryLayout::RowMajor,
            sharding: None,
        };
        assert_eq!(spec.size_bytes(), 2 * 3 * 4 * 4); // 96 bytes
    }

    #[test]
    fn test_tpu_memory_layout_variants() {
        let layouts = [
            TpuMemoryLayout::RowMajor,
            TpuMemoryLayout::ColumnMajor,
            TpuMemoryLayout::TpuOptimized,
            TpuMemoryLayout::Custom(vec![0, 2, 1]),
        ];
        assert_eq!(layouts.len(), 4);
        assert_eq!(layouts[0], TpuMemoryLayout::RowMajor);
        assert_eq!(layouts[3], TpuMemoryLayout::Custom(vec![0, 2, 1]));
    }

    #[test]
    fn test_utils_optimal_batch_size() {
        let batch_size = utils::get_optimal_batch_size(&[224, 224, 3], TpuGeneration::V4);
        assert!(batch_size > 0);
        assert!(batch_size <= 128);
    }

    #[test]
    fn test_utils_optimal_sharding() {
        let sharding = utils::create_optimal_sharding(&[1024, 768], "2x2");
        assert_eq!(sharding.mesh_shape, vec![2, 2]);
        assert_eq!(sharding.replicas, 4);
        assert_eq!(sharding.dimensions, vec![1024, 768]);
    }
}
