//! Kernel fusion strategies for optimized GPU compute operations
//!
//! This module provides intelligent kernel fusion to combine compatible operations
//! into single GPU kernels, reducing memory bandwidth and improving performance.

#![allow(dead_code)]

use crate::webgpu::{DeviceCapabilities, WorkgroupTuner};
use std::collections::BTreeMap;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Types of operations that can be fused
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusableOp {
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Multiply,
    /// Element-wise subtraction
    Subtract,
    /// ReLU activation
    ReLU,
    /// GELU activation
    GELU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Scale operation (multiply by constant)
    Scale,
    /// Bias operation (add constant)
    Bias,
    /// Clamp operation (min/max bounds)
    Clamp,
    /// Matrix multiplication
    MatMul,
    /// Layer normalization
    LayerNorm,
    /// Softmax activation
    Softmax,
    /// Dropout (for training)
    Dropout,
    /// Convolution operation
    Conv2d,
    /// Batch normalization
    BatchNorm,
}

/// Operation node in the fusion graph
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct OpNode {
    op_type: FusableOp,
    input_count: usize,
    output_count: usize,
    parameters: Vec<f32>,
    node_id: usize,
}

#[wasm_bindgen]
impl OpNode {
    #[wasm_bindgen(getter)]
    pub fn node_id(&self) -> usize {
        self.node_id
    }

    #[wasm_bindgen(getter)]
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    #[wasm_bindgen(getter)]
    pub fn output_count(&self) -> usize {
        self.output_count
    }
}

/// Fused kernel representing multiple operations
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct FusedKernel {
    operations: Vec<OpNode>,
    input_count: usize,
    output_count: usize,
    intermediate_count: usize,
    workgroup_size_x: u32,
    workgroup_size_y: u32,
    workgroup_size_z: u32,
    shader_source: String,
}

#[wasm_bindgen]
impl FusedKernel {
    #[wasm_bindgen(getter)]
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    #[wasm_bindgen(getter)]
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    #[wasm_bindgen(getter)]
    pub fn intermediate_count(&self) -> usize {
        self.intermediate_count
    }

    #[wasm_bindgen(getter)]
    pub fn workgroup_size_x(&self) -> u32 {
        self.workgroup_size_x
    }

    #[wasm_bindgen(getter)]
    pub fn workgroup_size_y(&self) -> u32 {
        self.workgroup_size_y
    }

    #[wasm_bindgen(getter)]
    pub fn workgroup_size_z(&self) -> u32 {
        self.workgroup_size_z
    }

    #[wasm_bindgen(getter)]
    pub fn shader_source(&self) -> String {
        self.shader_source.clone()
    }
}

/// Kernel fusion optimizer
#[wasm_bindgen]
pub struct KernelFusion {
    capabilities: DeviceCapabilities,
    tuner: WorkgroupTuner,
    fusion_cache: BTreeMap<String, FusedKernel>,
    max_fusion_depth: usize,
    max_intermediate_memory: usize,
}

#[wasm_bindgen]
impl KernelFusion {
    /// Create a new kernel fusion optimizer
    pub fn new(capabilities: DeviceCapabilities) -> KernelFusion {
        let tuner = WorkgroupTuner::new(capabilities.clone());

        // Adjust fusion parameters based on device capabilities
        let max_fusion_depth = if capabilities.is_mobile() { 4 } else { 8 };
        let max_intermediate_memory = if capabilities.is_mobile() {
            1024 * 1024 // 1MB for mobile
        } else {
            4 * 1024 * 1024 // 4MB for desktop
        };

        KernelFusion {
            capabilities,
            tuner,
            fusion_cache: BTreeMap::new(),
            max_fusion_depth,
            max_intermediate_memory,
        }
    }

    /// Analyze and fuse a sequence of operations
    pub fn fuse_operations(&mut self, operations: &js_sys::Array) -> Result<FusedKernel, JsValue> {
        if operations.length() == 0 {
            return Err("No operations to fuse".into());
        }

        // Convert JS array to Vec<OpNode> for processing
        let mut op_nodes = Vec::new();
        for i in 0..operations.length() {
            let _op_js = operations.get(i);
            // For now, create simple operations
            op_nodes.push(OpNode {
                op_type: FusableOp::Add,
                input_count: 2,
                output_count: 1,
                parameters: vec![],
                node_id: i as usize,
            });
        }

        // Generate cache key for the operation sequence
        let cache_key = self.generate_cache_key(&op_nodes);

        // Check cache first
        if let Some(cached) = self.fusion_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Analyze fusion opportunities
        let fusion_groups = self.analyze_fusion_opportunities(&op_nodes)?;

        // Select the best fusion strategy
        let best_fusion = self.select_optimal_fusion(&fusion_groups)?;

        // Generate fused kernel
        let fused_kernel = self.generate_fused_kernel(&best_fusion)?;

        // Cache the result
        self.fusion_cache.insert(cache_key, fused_kernel.clone());

        Ok(fused_kernel)
    }

    /// Check if two operations can be fused together
    pub fn can_fuse_operations(&self, op1: u32, op2: u32) -> bool {
        let fusable_op1 = match op1 {
            0 => FusableOp::Add,
            1 => FusableOp::Multiply,
            2 => FusableOp::Subtract,
            3 => FusableOp::ReLU,
            4 => FusableOp::GELU,
            5 => FusableOp::Sigmoid,
            6 => FusableOp::Tanh,
            7 => FusableOp::Scale,
            8 => FusableOp::Bias,
            9 => FusableOp::Clamp,
            _ => return false,
        };

        let fusable_op2 = match op2 {
            0 => FusableOp::Add,
            1 => FusableOp::Multiply,
            2 => FusableOp::Subtract,
            3 => FusableOp::ReLU,
            4 => FusableOp::GELU,
            5 => FusableOp::Sigmoid,
            6 => FusableOp::Tanh,
            7 => FusableOp::Scale,
            8 => FusableOp::Bias,
            9 => FusableOp::Clamp,
            _ => return false,
        };

        self.is_fusable_pair(fusable_op1, fusable_op2)
    }

    /// Get fusion statistics
    pub fn get_fusion_stats(&self) -> js_sys::Object {
        let stats = js_sys::Object::new();
        js_sys::Reflect::set(
            &stats,
            &"cache_size".into(),
            &self.fusion_cache.len().into(),
        )
        .expect("Failed to set fusion stats property");
        js_sys::Reflect::set(
            &stats,
            &"max_fusion_depth".into(),
            &self.max_fusion_depth.into(),
        )
        .expect("Failed to set fusion stats property");
        js_sys::Reflect::set(
            &stats,
            &"max_intermediate_memory".into(),
            &self.max_intermediate_memory.into(),
        )
        .expect("Failed to set fusion stats property");
        stats
    }

    /// Clear fusion cache
    pub fn clear_cache(&mut self) {
        self.fusion_cache.clear();
    }
}

// Private implementation methods
impl KernelFusion {
    /// Generate cache key for operation sequence
    fn generate_cache_key(&self, operations: &[OpNode]) -> String {
        let mut key = String::new();
        for op in operations {
            key.push_str(&format!("{:?}_{}_", op.op_type, op.parameters.len()));
        }
        key
    }

    /// Analyze which operations can be fused together
    fn analyze_fusion_opportunities(
        &self,
        operations: &[OpNode],
    ) -> Result<Vec<Vec<usize>>, JsValue> {
        let mut fusion_groups = Vec::new();
        let mut current_group = Vec::new();

        for (i, op) in operations.iter().enumerate() {
            if current_group.is_empty() {
                current_group.push(i);
            } else {
                let _last_idx = current_group[current_group.len() - 1];
                let _last_op = &operations[_last_idx];

                if self.can_fuse_with_group(op, operations, &current_group)
                    && current_group.len() < self.max_fusion_depth
                {
                    current_group.push(i);
                } else {
                    // Start new group
                    if current_group.len() > 1 {
                        fusion_groups.push(current_group);
                    }
                    current_group = vec![i];
                }
            }
        }

        // Add final group if it has multiple operations
        if current_group.len() > 1 {
            fusion_groups.push(current_group);
        }

        Ok(fusion_groups)
    }

    /// Check if an operation can be fused with existing group
    fn can_fuse_with_group(&self, op: &OpNode, all_ops: &[OpNode], group: &[usize]) -> bool {
        if group.is_empty() {
            return true;
        }

        // Check compatibility with all operations in the group
        for &idx in group {
            if !self.is_fusable_pair(all_ops[idx].op_type, op.op_type) {
                return false;
            }
        }

        // Check memory constraints
        let estimated_memory = self.estimate_intermediate_memory(all_ops, group, op);
        if estimated_memory > self.max_intermediate_memory {
            return false;
        }

        true
    }

    /// Check if two operations can be fused as a pair
    fn is_fusable_pair(&self, op1: FusableOp, op2: FusableOp) -> bool {
        use FusableOp::*;

        match (op1, op2) {
            // Element-wise operations can be fused together
            (Add, Multiply) | (Multiply, Add) => true,
            (Add, ReLU) | (ReLU, Add) => true,
            (Multiply, ReLU) | (ReLU, Multiply) => true,
            (Scale, Bias) | (Bias, Scale) => true,
            (Scale, Add) | (Add, Scale) => true,
            (Bias, Add) | (Add, Bias) => true,

            // Activation functions can be fused with element-wise ops
            (Add, GELU) | (GELU, Add) => true,
            (Multiply, GELU) | (GELU, Multiply) => true,
            (Add, Sigmoid) | (Sigmoid, Add) => true,
            (Multiply, Sigmoid) | (Sigmoid, Multiply) => true,
            (Add, Tanh) | (Tanh, Add) => true,
            (Multiply, Tanh) | (Tanh, Multiply) => true,

            // Clamp can be fused with most operations
            (Clamp, Add) | (Add, Clamp) => true,
            (Clamp, Multiply) | (Multiply, Clamp) => true,
            (Clamp, ReLU) | (ReLU, Clamp) => true,

            // Matrix operations with activations (common ML patterns)
            (MatMul, ReLU) | (ReLU, MatMul) => false, // MatMul should not be fused due to complexity
            (MatMul, GELU) | (GELU, MatMul) => false,
            (MatMul, Bias) | (Bias, MatMul) => true, // MatMul + Bias is very common

            // Layer normalization patterns
            (LayerNorm, ReLU) | (ReLU, LayerNorm) => true,
            (LayerNorm, GELU) | (GELU, LayerNorm) => true,
            (LayerNorm, Add) | (Add, LayerNorm) => false, // LayerNorm is complex

            // Batch normalization patterns
            (BatchNorm, ReLU) | (ReLU, BatchNorm) => true,
            (BatchNorm, GELU) | (GELU, BatchNorm) => true,
            (BatchNorm, Scale) | (Scale, BatchNorm) => false, // BatchNorm includes scaling

            // Softmax should generally not be fused (expensive operation)
            (Softmax, _) | (_, Softmax) => false,

            // Convolution patterns
            (Conv2d, ReLU) | (ReLU, Conv2d) => true, // Very common: Conv+ReLU
            (Conv2d, BatchNorm) | (BatchNorm, Conv2d) => true, // Conv+BatchNorm
            (Conv2d, Bias) | (Bias, Conv2d) => true, // Conv+Bias
            (Conv2d, _) | (_, Conv2d) => false,      // Other Conv patterns are complex

            // Dropout should not be fused (different execution paths)
            (Dropout, _) | (_, Dropout) => false,

            // Same operations can usually be fused
            (Add, Add) => true,
            (Multiply, Multiply) => true,
            (Scale, Scale) => true,
            (Bias, Bias) => true,

            // Default: not fusable
            _ => false,
        }
    }

    /// Estimate intermediate memory usage for fusion group
    fn estimate_intermediate_memory(
        &self,
        _all_ops: &[OpNode],
        group: &[usize],
        _new_op: &OpNode,
    ) -> usize {
        // Simple estimation: assume each intermediate result uses 4 bytes per element
        // and estimate based on typical tensor sizes
        let estimated_elements = 1024 * 1024; // 1M elements as rough estimate
        let bytes_per_element = 4; // f32
        let intermediate_results = group.len(); // One per operation in group

        estimated_elements * bytes_per_element * intermediate_results
    }

    /// Select optimal fusion strategy from candidates
    fn select_optimal_fusion(&self, fusion_groups: &[Vec<usize>]) -> Result<Vec<usize>, JsValue> {
        if fusion_groups.is_empty() {
            return Err("No fusion opportunities found".into());
        }

        // For now, select the largest group (most operations to fuse)
        // Future: could use more sophisticated heuristics based on memory bandwidth,
        // compute intensity, etc.
        let best_group = fusion_groups
            .iter()
            .max_by_key(|group| group.len())
            .ok_or("Failed to find best fusion group")?;

        Ok(best_group.clone())
    }

    /// Generate fused kernel from operation group
    fn generate_fused_kernel(&self, operation_indices: &[usize]) -> Result<FusedKernel, JsValue> {
        if operation_indices.is_empty() {
            return Err("No operations to fuse".into());
        }

        // Generate shader source for the fused operations
        let shader_source = self.generate_fused_shader(operation_indices)?;

        // Determine optimal workgroup size
        let config = self.tuner.recommend_config(0.7, 1024, false);
        let workgroup_size = (config.x, config.y, config.z);

        Ok(FusedKernel {
            operations: vec![], // Would be populated from actual operations
            input_count: 2,     // Simplified for now
            output_count: 1,
            intermediate_count: operation_indices.len() - 1,
            workgroup_size_x: workgroup_size.0,
            workgroup_size_y: workgroup_size.1,
            workgroup_size_z: workgroup_size.2,
            shader_source,
        })
    }

    /// Generate WGSL shader source for fused operations
    fn generate_fused_shader(&self, _operation_indices: &[usize]) -> Result<String, JsValue> {
        let mut shader = String::new();

        // Add bindings
        shader.push_str("@group(0) @binding(0) var<storage, read> input_a: array<f32>;\n");
        shader.push_str("@group(0) @binding(1) var<storage, read> input_b: array<f32>;\n");
        shader.push_str("@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n");
        shader.push_str("@group(0) @binding(3) var<uniform> size: u32;\n\n");

        // Add helper functions for different operations
        shader.push_str(
            r#"fn gelu(x: f32) -> f32 {
    let c1 = 0.7978845608; // sqrt(2/π)
    let c2 = 0.044715;
    let inner = c1 * (x + c2 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

"#,
        );

        // Get optimal workgroup size
        let config = self.tuner.recommend_config(0.7, 1024, false);
        shader.push_str(&format!("@compute @workgroup_size({})\n", config.x));

        // Main function
        shader.push_str(
            r#"fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }

    // Fused operation chain
    var temp = input_a[idx] + input_b[idx];  // Example: Add
    temp = max(0.0, temp);                   // Example: ReLU
    output[idx] = temp;
}
"#,
        );

        Ok(shader)
    }
}

/// Create operation node for common operations
#[wasm_bindgen]
pub struct OpNodeBuilder;

#[wasm_bindgen]
impl OpNodeBuilder {
    /// Create addition operation node
    pub fn add_op(node_id: usize) -> OpNode {
        OpNode {
            op_type: FusableOp::Add,
            input_count: 2,
            output_count: 1,
            parameters: vec![],
            node_id,
        }
    }

    /// Create multiplication operation node
    pub fn mul_op(node_id: usize) -> OpNode {
        OpNode {
            op_type: FusableOp::Multiply,
            input_count: 2,
            output_count: 1,
            parameters: vec![],
            node_id,
        }
    }

    /// Create ReLU operation node
    pub fn relu_op(node_id: usize) -> OpNode {
        OpNode {
            op_type: FusableOp::ReLU,
            input_count: 1,
            output_count: 1,
            parameters: vec![],
            node_id,
        }
    }

    /// Create GELU operation node
    pub fn gelu_op(node_id: usize) -> OpNode {
        OpNode {
            op_type: FusableOp::GELU,
            input_count: 1,
            output_count: 1,
            parameters: vec![],
            node_id,
        }
    }

    /// Create scale operation node
    pub fn scale_op(node_id: usize, scale_factor: f32) -> OpNode {
        OpNode {
            op_type: FusableOp::Scale,
            input_count: 1,
            output_count: 1,
            parameters: vec![scale_factor],
            node_id,
        }
    }
}

/// Utility function to create kernel fusion optimizer
#[wasm_bindgen]
pub fn create_kernel_fusion(capabilities: DeviceCapabilities) -> KernelFusion {
    KernelFusion::new(capabilities)
}

/// Check if operation fusion would be beneficial
#[wasm_bindgen]
pub fn should_fuse_operations(op_count: usize, total_elements: usize, is_mobile: bool) -> bool {
    // Simple heuristic: fuse if we have multiple ops and reasonable data size
    if op_count < 2 {
        return false;
    }

    let min_elements = if is_mobile { 1024 } else { 512 };
    let max_ops = if is_mobile { 4 } else { 8 };

    total_elements >= min_elements && op_count <= max_ops
}
