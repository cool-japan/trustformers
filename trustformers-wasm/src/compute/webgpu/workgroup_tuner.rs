//! Dynamic workgroup size tuning for optimal GPU performance
//!
//! This module analyzes device capabilities and generates optimized
//! compute shaders with tuned workgroup sizes for different operations.

use crate::webgpu::DeviceCapabilities;
use std::collections::BTreeMap;
use std::format;
use std::string::{String, ToString};
use wasm_bindgen::prelude::*;

/// Operation types for workgroup tuning
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperationType {
    /// Matrix multiplication (2D workgroups)
    MatMul,
    /// Element-wise operations (1D workgroups)
    ElementWise,
    /// Reduction operations (requires shared memory)
    Reduction,
    /// Attention mechanisms (mixed 1D/2D)
    Attention,
    /// Convolution operations
    Convolution,
}

/// Workgroup configuration for different operation types
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct WorkgroupConfig {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub shared_memory_size: u32,
}

#[wasm_bindgen]
impl WorkgroupConfig {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> u32 {
        self.x
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> u32 {
        self.y
    }

    #[wasm_bindgen(getter)]
    pub fn z(&self) -> u32 {
        self.z
    }

    #[wasm_bindgen(getter)]
    pub fn shared_memory_size(&self) -> u32 {
        self.shared_memory_size
    }
}

impl WorkgroupConfig {
    pub fn new_1d(size: u32) -> Self {
        Self {
            x: size,
            y: 1,
            z: 1,
            shared_memory_size: 0,
        }
    }

    pub fn new_2d(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            z: 1,
            shared_memory_size: 0,
        }
    }

    pub fn with_shared_memory(mut self, size: u32) -> Self {
        self.shared_memory_size = size;
        self
    }
}

/// Workgroup size tuner that optimizes based on device capabilities
#[wasm_bindgen]
pub struct WorkgroupTuner {
    capabilities: DeviceCapabilities,
    configs: BTreeMap<OperationType, WorkgroupConfig>,
    performance_profile: PerformanceProfile,
}

#[derive(Debug, Clone)]
struct PerformanceProfile {
    preferred_1d_size: u32,
    preferred_2d_x: u32,
    preferred_2d_y: u32,
    max_shared_memory: u32,
    memory_bandwidth_tier: u32,
}

#[wasm_bindgen]
impl WorkgroupTuner {
    /// Create a new workgroup tuner with device capabilities
    pub fn new(capabilities: DeviceCapabilities) -> WorkgroupTuner {
        let performance_profile = Self::analyze_performance_profile(&capabilities);
        let mut tuner = WorkgroupTuner {
            capabilities,
            configs: BTreeMap::new(),
            performance_profile,
        };

        // Generate optimal configurations for all operation types
        tuner.generate_optimal_configs();
        tuner
    }

    /// Get workgroup configuration for a specific operation type (private)
    fn get_config_internal(&self, op_type: OperationType) -> Option<WorkgroupConfig> {
        self.configs.get(&op_type).copied()
    }

    /// Generate optimized shader with tuned workgroup size (private)
    fn generate_shader_internal(&self, op_type: OperationType, base_shader: &str) -> String {
        if let Some(config) = self.get_config_internal(op_type) {
            self.apply_workgroup_config(base_shader, config)
        } else {
            base_shader.to_string()
        }
    }

    /// Get workgroup configuration for a specific operation type
    pub fn get_config(&self, op_type: OperationType) -> Option<WorkgroupConfig> {
        self.get_config_internal(op_type)
    }

    /// Generate optimized shader with tuned workgroup size
    pub fn generate_shader(&self, op_type: OperationType, base_shader: &str) -> String {
        self.generate_shader_internal(op_type, base_shader)
    }

    /// Get recommended configuration for custom operations
    pub fn recommend_config(
        &self,
        operation_complexity: f32,
        memory_usage: u32,
        is_2d: bool,
    ) -> WorkgroupConfig {
        if is_2d {
            let base_x = self.performance_profile.preferred_2d_x;
            let base_y = self.performance_profile.preferred_2d_y;

            // Adjust based on complexity
            let scale_factor = if operation_complexity > 0.8 { 0.75 } else { 1.0 };
            let x = ((base_x as f32 * scale_factor) as u32).clamp(4, 32);
            let y = ((base_y as f32 * scale_factor) as u32).clamp(4, 32);

            WorkgroupConfig::new_2d(x, y)
                .with_shared_memory(memory_usage.min(self.performance_profile.max_shared_memory))
        } else {
            let base_size = self.performance_profile.preferred_1d_size;
            let scale_factor = if operation_complexity > 0.8 { 1.5 } else { 1.0 };
            let size = ((base_size as f32 * scale_factor) as u32)
                .max(64)
                .min(self.capabilities.max_compute_workgroup_size());

            WorkgroupConfig::new_1d(size)
                .with_shared_memory(memory_usage.min(self.performance_profile.max_shared_memory))
        }
    }
}

// Private implementation methods
impl WorkgroupTuner {
    /// Analyze device capabilities to create performance profile
    fn analyze_performance_profile(caps: &DeviceCapabilities) -> PerformanceProfile {
        let is_mobile = caps.is_mobile();
        let performance_tier = caps.performance_tier();
        let max_workgroup_size = caps.max_compute_workgroup_size();

        // Base configurations based on device type
        let (base_1d, base_2d_x, _base_2d_y) = if is_mobile {
            // Conservative sizes for mobile devices
            (128, 8, 8)
        } else {
            // More aggressive sizes for desktop/high-performance devices
            (256, 16, 16)
        };

        // Adjust based on performance tier
        let tier_multiplier = match performance_tier {
            1 => 0.5,  // Low-end: reduce workgroup sizes
            2 => 0.75, // Mid-range: slightly reduce
            3 => 1.0,  // High-end: use base sizes
            4 => 1.25, // Ultra-high-end: increase sizes
            _ => 1.0,
        };

        let preferred_1d =
            ((base_1d as f32 * tier_multiplier) as u32).clamp(64, max_workgroup_size);

        let preferred_2d_x = ((base_2d_x as f32 * tier_multiplier) as u32)
            .max(4)
            .min((max_workgroup_size as f32).sqrt() as u32);

        let preferred_2d_y = preferred_2d_x; // Keep square for simplicity

        // Estimate shared memory limits (conservative)
        let max_shared_memory = if is_mobile { 8192 } else { 16384 };

        // Memory bandwidth tier based on GPU memory and performance
        let memory_bandwidth_tier =
            if caps.gpu_memory_limit() > 2_000_000_000.0 && performance_tier >= 3 {
                3 // High bandwidth
            } else if caps.gpu_memory_limit() > 500_000_000.0 {
                2 // Medium bandwidth
            } else {
                1 // Low bandwidth
            };

        PerformanceProfile {
            preferred_1d_size: preferred_1d,
            preferred_2d_x,
            preferred_2d_y,
            max_shared_memory,
            memory_bandwidth_tier,
        }
    }

    /// Generate optimal configurations for all operation types
    fn generate_optimal_configs(&mut self) {
        // Matrix multiplication: 2D workgroups with shared memory
        let matmul_shared_memory = if self.performance_profile.memory_bandwidth_tier >= 2 {
            2048 // Use shared memory for better performance
        } else {
            0 // Skip shared memory on low-bandwidth devices
        };

        self.configs.insert(
            OperationType::MatMul,
            WorkgroupConfig::new_2d(
                self.performance_profile.preferred_2d_x,
                self.performance_profile.preferred_2d_y,
            )
            .with_shared_memory(matmul_shared_memory),
        );

        // Element-wise operations: 1D workgroups, larger sizes for bandwidth utilization
        let elementwise_size = if self.performance_profile.memory_bandwidth_tier >= 2 {
            self.performance_profile.preferred_1d_size * 2
        } else {
            self.performance_profile.preferred_1d_size
        }
        .min(self.capabilities.max_compute_workgroup_size());

        self.configs.insert(
            OperationType::ElementWise,
            WorkgroupConfig::new_1d(elementwise_size),
        );

        // Reduction operations: Smaller workgroups with shared memory
        let reduction_size = (self.performance_profile.preferred_1d_size / 2).max(64);
        self.configs.insert(
            OperationType::Reduction,
            WorkgroupConfig::new_1d(reduction_size)
                .with_shared_memory(self.performance_profile.max_shared_memory / 2),
        );

        // Attention mechanisms: Balanced 2D configuration
        let attention_x = (self.performance_profile.preferred_2d_x * 3 / 4).max(4);
        let attention_y = (self.performance_profile.preferred_2d_y * 3 / 4).max(4);
        self.configs.insert(
            OperationType::Attention,
            WorkgroupConfig::new_2d(attention_x, attention_y).with_shared_memory(1024),
        );

        // Convolution: Square workgroups optimized for spatial operations
        let conv_size =
            (self.performance_profile.preferred_2d_x + self.performance_profile.preferred_2d_y) / 2;
        self.configs.insert(
            OperationType::Convolution,
            WorkgroupConfig::new_2d(conv_size, conv_size),
        );
    }

    /// Apply workgroup configuration to shader source
    fn apply_workgroup_config(&self, shader: &str, config: WorkgroupConfig) -> String {
        // Replace existing @workgroup_size directives
        let workgroup_directive = if config.y == 1 && config.z == 1 {
            format!("@workgroup_size({})", config.x)
        } else if config.z == 1 {
            format!("@workgroup_size({}, {})", config.x, config.y)
        } else {
            format!("@workgroup_size({}, {}, {})", config.x, config.y, config.z)
        };

        // Simple regex-like replacement for @workgroup_size
        let lines: std::vec::Vec<&str> = shader.lines().collect();
        let mut result = String::new();

        for line in lines {
            if line.trim_start().starts_with("@workgroup_size") {
                // Replace the workgroup_size line
                let indentation = line.len() - line.trim_start().len();
                result.push_str(&" ".repeat(indentation));
                result.push_str(&workgroup_directive);
                result.push('\n');
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }

        // Add shared memory if needed
        if config.shared_memory_size > 0 {
            result = self.add_shared_memory_to_shader(&result, config.shared_memory_size);
        }

        result
    }

    /// Add shared memory declarations to shader
    fn add_shared_memory_to_shader(&self, shader: &str, shared_size: u32) -> String {
        // Add shared memory declaration after the bindings
        let shared_memory_decl = format!(
            "var<workgroup> shared_memory: array<f32, {}>;",
            shared_size / 4 // Convert bytes to f32 elements
        );

        // Find a good place to insert shared memory (after last binding)
        let lines: std::vec::Vec<&str> = shader.lines().collect();
        let mut result = String::new();
        let mut inserted = false;

        for line in lines {
            result.push_str(line);
            result.push('\n');

            // Insert shared memory after the last binding declaration
            if !inserted && (line.contains("@binding") || line.contains("@group")) {
                // Look ahead to see if there are more bindings
                let remaining = shader[shader.find(line).unwrap() + line.len()..].to_string();
                if !remaining.contains("@binding") && !remaining.contains("@group") {
                    result.push('\n');
                    result.push_str(&shared_memory_decl);
                    result.push('\n');
                    inserted = true;
                }
            }
        }

        result
    }
}

/// Utility function to create workgroup tuner from device capabilities
#[wasm_bindgen]
pub fn create_workgroup_tuner(capabilities: DeviceCapabilities) -> WorkgroupTuner {
    WorkgroupTuner::new(capabilities)
}

/// Get optimal workgroup size for matrix multiplication
#[wasm_bindgen]
pub fn get_optimal_matmul_workgroup(capabilities: DeviceCapabilities) -> js_sys::Array {
    let tuner = WorkgroupTuner::new(capabilities);
    if let Some(config) = tuner.get_config_internal(OperationType::MatMul) {
        let result = js_sys::Array::new();
        result.push(&JsValue::from(config.x));
        result.push(&JsValue::from(config.y));
        result
    } else {
        let result = js_sys::Array::new();
        result.push(&JsValue::from(8u32));
        result.push(&JsValue::from(8u32));
        result
    }
}

/// Get optimal workgroup size for element-wise operations
#[wasm_bindgen]
pub fn get_optimal_elementwise_workgroup(capabilities: DeviceCapabilities) -> u32 {
    let tuner = WorkgroupTuner::new(capabilities);
    tuner
        .get_config_internal(OperationType::ElementWise)
        .map(|config| config.x)
        .unwrap_or(256)
}
