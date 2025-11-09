//! Enhanced shader manager with dynamic workgroup tuning
//!
//! This module provides optimized WGSL shaders with device-specific workgroup tuning

use crate::webgpu::{DeviceCapabilities, OperationType, WorkgroupTuner};
use std::collections::BTreeMap;
use std::string::{String, ToString};
use wasm_bindgen::prelude::*;

/// Enhanced shader manager with workgroup optimization
#[wasm_bindgen]
pub struct ShaderManager {
    tuner: WorkgroupTuner,
    cached_shaders: BTreeMap<String, String>,
}

#[wasm_bindgen]
impl ShaderManager {
    /// Create a new shader manager with device-optimized workgroup tuning
    pub fn new(capabilities: DeviceCapabilities) -> ShaderManager {
        ShaderManager {
            tuner: WorkgroupTuner::new(capabilities),
            cached_shaders: BTreeMap::new(),
        }
    }

    /// Get optimized matrix multiplication shader
    pub fn get_matmul_shader(&mut self) -> String {
        let cache_key = "matmul_optimized".to_string();

        if let Some(cached) = self.cached_shaders.get(&cache_key) {
            return cached.clone();
        }

        let base_shader = Self::get_base_matmul_shader();
        let optimized = self.tuner.generate_shader(OperationType::MatMul, &base_shader);
        self.cached_shaders.insert(cache_key, optimized.clone());
        optimized
    }

    /// Get optimized element-wise addition shader
    pub fn get_add_shader(&mut self) -> String {
        let cache_key = "add_optimized".to_string();

        if let Some(cached) = self.cached_shaders.get(&cache_key) {
            return cached.clone();
        }

        let base_shader = Self::get_base_add_shader();
        let optimized = self.tuner.generate_shader(OperationType::ElementWise, &base_shader);
        self.cached_shaders.insert(cache_key, optimized.clone());
        optimized
    }

    /// Get optimized element-wise multiplication shader
    pub fn get_mul_shader(&mut self) -> String {
        let cache_key = "mul_optimized".to_string();

        if let Some(cached) = self.cached_shaders.get(&cache_key) {
            return cached.clone();
        }

        let base_shader = Self::get_base_mul_shader();
        let optimized = self.tuner.generate_shader(OperationType::ElementWise, &base_shader);
        self.cached_shaders.insert(cache_key, optimized.clone());
        optimized
    }

    /// Get optimized ReLU activation shader
    pub fn get_relu_shader(&mut self) -> String {
        let cache_key = "relu_optimized".to_string();

        if let Some(cached) = self.cached_shaders.get(&cache_key) {
            return cached.clone();
        }

        let base_shader = Self::get_base_relu_shader();
        let optimized = self.tuner.generate_shader(OperationType::ElementWise, &base_shader);
        self.cached_shaders.insert(cache_key, optimized.clone());
        optimized
    }

    /// Get optimized GELU activation shader
    pub fn get_gelu_shader(&mut self) -> String {
        let cache_key = "gelu_optimized".to_string();

        if let Some(cached) = self.cached_shaders.get(&cache_key) {
            return cached.clone();
        }

        let base_shader = Self::get_base_gelu_shader();
        let optimized = self.tuner.generate_shader(OperationType::ElementWise, &base_shader);
        self.cached_shaders.insert(cache_key, optimized.clone());
        optimized
    }

    /// Get optimized softmax shader
    pub fn get_softmax_shader(&mut self) -> String {
        let cache_key = "softmax_optimized".to_string();

        if let Some(cached) = self.cached_shaders.get(&cache_key) {
            return cached.clone();
        }

        let base_shader = Self::get_base_softmax_shader();
        let optimized = self.tuner.generate_shader(OperationType::Reduction, &base_shader);
        self.cached_shaders.insert(cache_key, optimized.clone());
        optimized
    }

    /// Get optimized layer normalization shader
    pub fn get_layer_norm_shader(&mut self) -> String {
        let cache_key = "layer_norm_optimized".to_string();

        if let Some(cached) = self.cached_shaders.get(&cache_key) {
            return cached.clone();
        }

        let base_shader = Self::get_base_layer_norm_shader();
        let optimized = self.tuner.generate_shader(OperationType::Reduction, &base_shader);
        self.cached_shaders.insert(cache_key, optimized.clone());
        optimized
    }

    /// Clear shader cache
    pub fn clear_cache(&mut self) {
        self.cached_shaders.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cached_shaders.len()
    }
}

// Base shader definitions (private methods)
impl ShaderManager {
    fn get_base_matmul_shader() -> String {
        r#"@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // M, N, K

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        let a_idx = row * K + k;
        let b_idx = k * N + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let result_idx = row * N + col;
    result[result_idx] = sum;
}"#
        .to_string()
    }

    fn get_base_add_shader() -> String {
        r#"@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    result[idx] = a[idx] + b[idx];
}"#
        .to_string()
    }

    fn get_base_mul_shader() -> String {
        r#"@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    result[idx] = a[idx] * b[idx];
}"#
        .to_string()
    }

    fn get_base_relu_shader() -> String {
        r#"@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    output[idx] = max(0.0, input[idx]);
}"#
        .to_string()
    }

    fn get_base_gelu_shader() -> String {
        r#"@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

fn gelu(x: f32) -> f32 {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let c1 = 0.7978845608; // sqrt(2/π)
    let c2 = 0.044715;
    let inner = c1 * (x + c2 * x * x * x);
    let tanh_inner = tanh(inner);
    return 0.5 * x * (1.0 + tanh_inner);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }
    output[idx] = gelu(input[idx]);
}"#
        .to_string()
    }

    fn get_base_softmax_shader() -> String {
        r#"@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>; // batch_size, feature_size

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let feature_idx = global_id.y;
    let batch_size = dims.x;
    let feature_size = dims.y;

    if (batch_idx >= batch_size) {
        return;
    }

    let base_idx = batch_idx * feature_size;

    // Find max for numerical stability
    var max_val = input[base_idx];
    for (var i = 1u; i < feature_size; i = i + 1u) {
        max_val = max(max_val, input[base_idx + i]);
    }

    // Compute exp and sum
    var sum = 0.0;
    for (var i = 0u; i < feature_size; i = i + 1u) {
        let exp_val = exp(input[base_idx + i] - max_val);
        output[base_idx + i] = exp_val;
        sum = sum + exp_val;
    }

    // Normalize
    for (var i = 0u; i < feature_size; i = i + 1u) {
        output[base_idx + i] = output[base_idx + i] / sum;
    }
}"#
        .to_string()
    }

    fn get_base_layer_norm_shader() -> String {
        r#"@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec2<u32>; // batch_size, feature_size

@compute @workgroup_size(1, 256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let feature_idx = global_id.y;
    let batch_size = dims.x;
    let feature_size = dims.y;

    if (batch_idx >= batch_size) {
        return;
    }

    let base_idx = batch_idx * feature_size;

    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < feature_size; i = i + 1u) {
        sum = sum + input[base_idx + i];
    }
    let mean = sum / f32(feature_size);

    // Compute variance
    var var_sum = 0.0;
    for (var i = 0u; i < feature_size; i = i + 1u) {
        let diff = input[base_idx + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f32(feature_size);
    let std_dev = sqrt(variance + 1e-5);

    // Normalize and apply scale/shift
    for (var i = 0u; i < feature_size; i = i + 1u) {
        let normalized = (input[base_idx + i] - mean) / std_dev;
        output[base_idx + i] = gamma[i] * normalized + beta[i];
    }
}"#
        .to_string()
    }
}

/// Utility function to create shader manager
#[wasm_bindgen]
pub fn create_shader_manager(capabilities: DeviceCapabilities) -> ShaderManager {
    ShaderManager::new(capabilities)
}
