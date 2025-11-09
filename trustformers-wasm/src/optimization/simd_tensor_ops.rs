//! SIMD-optimized tensor operations for WebAssembly
//!
//! This module provides highly optimized tensor operations using WebAssembly SIMD instructions
//! for improved performance in browser environments.

use crate::core::tensor::WasmTensor;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// SIMD-enhanced tensor operations
#[wasm_bindgen]
pub struct SimdTensorOps;

#[wasm_bindgen]
impl SimdTensorOps {
    /// Create a new SIMD tensor operations instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        SimdTensorOps
    }

    /// Check if SIMD is available
    pub fn is_simd_available() -> bool {
        cfg!(target_feature = "simd128")
    }

    /// Element-wise addition using SIMD acceleration
    pub fn add_tensors(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if a.shape() != b.shape() {
            return Err(JsValue::from_str(
                "Tensors must have the same shape for addition",
            ));
        }

        let a_data = a.data();
        let b_data = b.data();
        let result;

        #[cfg(target_arch = "wasm32")]
        {
            if cfg!(target_feature = "simd128") {
                result = self.simd_add(&a_data, &b_data);
            } else {
                // Fallback to scalar operations
                result = self.scalar_add(&a_data, &b_data);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            result = self.scalar_add(&a_data, &b_data);
        }

        WasmTensor::new(result, a.shape())
    }

    /// Element-wise multiplication using SIMD acceleration
    pub fn multiply_tensors(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if a.shape() != b.shape() {
            return Err(JsValue::from_str(
                "Tensors must have the same shape for multiplication",
            ));
        }

        let a_data = a.data();
        let b_data = b.data();
        let result;

        #[cfg(target_arch = "wasm32")]
        {
            if cfg!(target_feature = "simd128") {
                result = self.simd_multiply(&a_data, &b_data);
            } else {
                result = self.scalar_multiply(&a_data, &b_data);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            result = self.scalar_multiply(&a_data, &b_data);
        }

        WasmTensor::new(result, a.shape())
    }

    /// Apply ReLU activation function using SIMD
    pub fn relu(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let data = input.data();
        let result;

        #[cfg(target_arch = "wasm32")]
        {
            if cfg!(target_feature = "simd128") {
                result = self.simd_relu(&data);
            } else {
                result = self.scalar_relu(&data);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            result = self.scalar_relu(&data);
        }

        WasmTensor::new(result, input.shape())
    }

    /// Apply softmax activation function using SIMD
    pub fn softmax(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let data = input.data();
        let shape = input.shape();

        // For simplicity, apply softmax across the last dimension
        if shape.is_empty() {
            return Err(JsValue::from_str("Cannot apply softmax to empty tensor"));
        }

        let last_dim = *shape.last().unwrap();
        if last_dim == 0 {
            return Err(JsValue::from_str("Last dimension cannot be zero"));
        }

        let mut result = Vec::with_capacity(data.len());

        // Apply softmax in chunks of last_dim size
        for chunk in data.chunks(last_dim) {
            let softmax_chunk = self.compute_softmax_chunk(chunk);
            result.extend(softmax_chunk);
        }

        WasmTensor::new(result, shape)
    }

    /// Compute matrix multiplication using optimized algorithms
    pub fn matmul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Basic 2D matrix multiplication validation
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(JsValue::from_str(
                "Matrix multiplication requires 2D tensors",
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(JsValue::from_str(
                "Matrix inner dimensions must match for multiplication",
            ));
        }

        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        let a_data = a.data();
        let b_data = b.data();
        let mut result = vec![0.0f32; m * n];

        // Use cache-friendly matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;

                #[cfg(target_arch = "wasm32")]
                {
                    if cfg!(target_feature = "simd128") {
                        sum = self.simd_dot_product(
                            &a_data[i * k..(i + 1) * k],
                            &self.get_column(&b_data, j, k, n),
                            k,
                        );
                    } else {
                        for l in 0..k {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                    }
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    for l in 0..k {
                        sum += a_data[i * k + l] * b_data[l * n + j];
                    }
                }

                result[i * n + j] = sum;
            }
        }

        WasmTensor::new(result, vec![m, n])
    }
}

impl SimdTensorOps {
    #[cfg(target_arch = "wasm32")]
    fn simd_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;

        // Process 4 elements at a time using SIMD
        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = v128_load(&a[idx..idx + 4] as *const [f32] as *const v128);
            let b_vec = v128_load(&b[idx..idx + 4] as *const [f32] as *const v128);
            let sum = f32x4_add(a_vec, b_vec);

            let mut temp = [0.0f32; 4];
            v128_store(&mut temp as *mut [f32] as *mut v128, sum);
            result.extend_from_slice(&temp);
        }

        // Handle remaining elements
        for i in (chunks * 4)..a.len() {
            result.push(a[i] + b[i]);
        }

        result
    }

    fn scalar_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    #[cfg(target_arch = "wasm32")]
    fn simd_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = v128_load(&a[idx..idx + 4] as *const [f32] as *const v128);
            let b_vec = v128_load(&b[idx..idx + 4] as *const [f32] as *const v128);
            let prod = f32x4_mul(a_vec, b_vec);

            let mut temp = [0.0f32; 4];
            v128_store(&mut temp as *mut [f32] as *mut v128, prod);
            result.extend_from_slice(&temp);
        }

        for i in (chunks * 4)..a.len() {
            result.push(a[i] * b[i]);
        }

        result
    }

    fn scalar_multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    #[cfg(target_arch = "wasm32")]
    fn simd_relu(&self, data: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(data.len());
        let chunks = data.len() / 4;
        let zero_vec = f32x4_splat(0.0);

        for i in 0..chunks {
            let idx = i * 4;
            let data_vec = v128_load(&data[idx..idx + 4] as *const [f32] as *const v128);
            let relu_vec = f32x4_max(data_vec, zero_vec);

            let mut temp = [0.0f32; 4];
            v128_store(&mut temp as *mut [f32] as *mut v128, relu_vec);
            result.extend_from_slice(&temp);
        }

        for i in (chunks * 4)..data.len() {
            result.push(data[i].max(0.0));
        }

        result
    }

    fn scalar_relu(&self, data: &[f32]) -> Vec<f32> {
        data.iter().map(|&x| x.max(0.0)).collect()
    }

    fn compute_softmax_chunk(&self, chunk: &[f32]) -> Vec<f32> {
        if chunk.is_empty() {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_val = chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) and sum
        let exp_values: Vec<f32> = chunk.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();

        // Normalize
        if sum > 0.0 {
            exp_values.into_iter().map(|x| x / sum).collect()
        } else {
            // Fallback: uniform distribution
            let uniform_val = 1.0 / chunk.len() as f32;
            vec![uniform_val; chunk.len()]
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn simd_dot_product(&self, a: &[f32], b: &[f32], len: usize) -> f32 {
        let mut sum = 0.0f32;
        let chunks = len / 4;
        let mut simd_sum = f32x4_splat(0.0);

        for i in 0..chunks {
            let idx = i * 4;
            if idx + 4 <= a.len() && idx + 4 <= b.len() {
                let a_vec = v128_load(&a[idx..idx + 4] as *const [f32] as *const v128);
                let b_vec = v128_load(&b[idx..idx + 4] as *const [f32] as *const v128);
                let prod = f32x4_mul(a_vec, b_vec);
                simd_sum = f32x4_add(simd_sum, prod);
            }
        }

        // Extract components and sum
        let mut temp = [0.0f32; 4];
        v128_store(&mut temp as *mut [f32] as *mut v128, simd_sum);
        sum += temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for i in (chunks * 4)..len.min(a.len()).min(b.len()) {
            sum += a[i] * b[i];
        }

        sum
    }

    #[allow(dead_code)]
    fn get_column(&self, matrix: &[f32], col: usize, rows: usize, cols: usize) -> Vec<f32> {
        let mut column = Vec::with_capacity(rows);
        for row in 0..rows {
            if row * cols + col < matrix.len() {
                column.push(matrix[row * cols + col]);
            } else {
                column.push(0.0);
            }
        }
        column
    }
}

impl Default for SimdTensorOps {
    fn default() -> Self {
        Self::new()
    }
}
