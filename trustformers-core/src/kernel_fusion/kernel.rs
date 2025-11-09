//! Fused kernel representation and implementation
//!
//! This module defines structures for representing fused kernels and their
//! various backend implementations.

use crate::kernel_fusion::graph::TensorInfo;
use crate::kernel_fusion::operation_types::FusionPattern;

/// Fused kernel representation
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub id: String,
    pub name: String,
    pub pattern: FusionPattern,
    pub operations: Vec<String>, // Original operation IDs
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    pub estimated_speedup: f64,
    pub memory_savings: usize,
    pub implementation: KernelImplementation,
}

#[derive(Debug, Clone)]
pub enum KernelImplementation {
    CUDA(String),   // CUDA kernel code
    ROCm(String),   // ROCm/HIP kernel code
    OpenCL(String), // OpenCL kernel code
    CPU(String),    // CPU implementation
    Vulkan(String), // Vulkan compute shader
    Metal(String),  // Metal compute shader
    WebGPU(String), // WebGPU shader
    SIMD(String),   // SIMD intrinsics
    ASIC(String),   // ASIC-specific kernel code
}

impl FusedKernel {
    pub fn new(id: String, name: String, pattern: FusionPattern, operations: Vec<String>) -> Self {
        Self {
            id,
            name,
            pattern,
            operations,
            inputs: Vec::new(),
            outputs: Vec::new(),
            estimated_speedup: 1.0,
            memory_savings: 0,
            implementation: KernelImplementation::CPU("".to_string()),
        }
    }

    pub fn with_inputs(mut self, inputs: Vec<TensorInfo>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<TensorInfo>) -> Self {
        self.outputs = outputs;
        self
    }

    pub fn with_speedup(mut self, speedup: f64) -> Self {
        self.estimated_speedup = speedup;
        self
    }

    pub fn with_memory_savings(mut self, savings: usize) -> Self {
        self.memory_savings = savings;
        self
    }

    pub fn with_implementation(mut self, implementation: KernelImplementation) -> Self {
        self.implementation = implementation;
        self
    }
}

impl KernelImplementation {
    pub fn platform(&self) -> &'static str {
        match self {
            KernelImplementation::CUDA(_) => "CUDA",
            KernelImplementation::ROCm(_) => "ROCm",
            KernelImplementation::OpenCL(_) => "OpenCL",
            KernelImplementation::CPU(_) => "CPU",
            KernelImplementation::Vulkan(_) => "Vulkan",
            KernelImplementation::Metal(_) => "Metal",
            KernelImplementation::WebGPU(_) => "WebGPU",
            KernelImplementation::SIMD(_) => "SIMD",
            KernelImplementation::ASIC(_) => "ASIC",
        }
    }

    pub fn code(&self) -> &str {
        match self {
            KernelImplementation::CUDA(code)
            | KernelImplementation::ROCm(code)
            | KernelImplementation::OpenCL(code)
            | KernelImplementation::CPU(code)
            | KernelImplementation::Vulkan(code)
            | KernelImplementation::Metal(code)
            | KernelImplementation::WebGPU(code)
            | KernelImplementation::SIMD(code)
            | KernelImplementation::ASIC(code) => code,
        }
    }
}
