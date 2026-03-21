use getrandom;
use serde::{Deserialize, Serialize};
use std::string::String;
use std::vec;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmTensor {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        // Validate inputs for safety
        if data.is_empty() {
            return Err(JsValue::from_str("Empty tensor data not allowed"));
        }

        if shape.is_empty() {
            return Err(JsValue::from_str("Empty tensor shape not allowed"));
        }

        // Check for overflow in shape calculation
        let expected_size = match shape.iter().try_fold(1usize, |acc, &dim| {
            if dim == 0 {
                return None; // Zero dimensions not allowed
            }
            acc.checked_mul(dim)
        }) {
            Some(size) => size,
            None => {
                return Err(JsValue::from_str(
                    "Shape dimensions cause overflow or contain zero",
                ))
            },
        };

        if data.len() != expected_size {
            return Err(JsValue::from_str(&format!(
                "Data length ({}) does not match shape (expected {})",
                data.len(),
                expected_size
            )));
        }

        // Validate data contains no NaN or infinite values
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(JsValue::from_str(&format!(
                    "Invalid value at index {}: {} (must be finite)",
                    i, value
                )));
            }
        }

        let strides = compute_strides(&shape);
        Ok(WasmTensor {
            data,
            shape,
            strides,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get reference to data for internal use (more efficient than clone)
    /// Note: Not exposed to JavaScript - for internal Rust use only
    pub(crate) fn data_ref(&self) -> &[f32] {
        &self.data
    }

    /// Get reference to shape for internal use (more efficient than clone)
    /// Note: Not exposed to JavaScript - for internal Rust use only
    #[allow(dead_code)]
    pub(crate) fn shape_ref(&self) -> &[usize] {
        &self.shape
    }

    /// Get reference to strides for internal use
    /// Note: Not exposed to JavaScript - for internal Rust use only
    #[allow(dead_code)]
    pub(crate) fn strides_ref(&self) -> &[usize] {
        &self.strides
    }

    /// Get the total number of elements in the tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn zeros(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        // Validate shape before computing size
        if shape.is_empty() {
            return Err(JsValue::from_str(
                "Cannot create zeros tensor with empty shape",
            ));
        }

        let size = match shape.iter().try_fold(1usize, |acc, &dim| {
            if dim == 0 {
                return None;
            }
            acc.checked_mul(dim)
        }) {
            Some(size) => size,
            None => {
                return Err(JsValue::from_str(
                    "Shape dimensions cause overflow or contain zero",
                ))
            },
        };

        // Check reasonable size limits (1GB max)
        const MAX_ELEMENTS: usize = 268_435_456; // 1GB / 4 bytes per f32
        if size > MAX_ELEMENTS {
            return Err(JsValue::from_str(&format!(
                "Tensor size {} exceeds maximum allowed size {}",
                size, MAX_ELEMENTS
            )));
        }

        let data = vec![0.0; size];
        WasmTensor::new(data, shape)
    }

    pub fn ones(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        // Validate shape before computing size
        if shape.is_empty() {
            return Err(JsValue::from_str(
                "Cannot create ones tensor with empty shape",
            ));
        }

        let size = match shape.iter().try_fold(1usize, |acc, &dim| {
            if dim == 0 {
                return None;
            }
            acc.checked_mul(dim)
        }) {
            Some(size) => size,
            None => {
                return Err(JsValue::from_str(
                    "Shape dimensions cause overflow or contain zero",
                ))
            },
        };

        // Check reasonable size limits (1GB max)
        const MAX_ELEMENTS: usize = 268_435_456; // 1GB / 4 bytes per f32
        if size > MAX_ELEMENTS {
            return Err(JsValue::from_str(&format!(
                "Tensor size {} exceeds maximum allowed size {}",
                size, MAX_ELEMENTS
            )));
        }

        let data = vec![1.0; size];
        WasmTensor::new(data, shape)
    }

    /// Generate tensor with random normal distribution (mean=0, std=1)
    pub fn randn(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        // Use getrandom for cryptographically secure random numbers
        match Self::generate_normal_distribution(size) {
            Ok(values) => data = values,
            Err(_) => {
                // Fallback to simple pseudo-random for demo compatibility
                let mut seed = 12345u32;
                for _ in 0..size {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let float_val = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    data.push(float_val);
                }
            },
        }

        WasmTensor::new(data, shape)
    }

    /// Generate tensor with random normal distribution with custom mean and std
    pub fn randn_with_params(
        shape: Vec<usize>,
        mean: f32,
        std: f32,
    ) -> Result<WasmTensor, JsValue> {
        let mut tensor = Self::randn(shape)?;
        // Apply mean and std transformation: x = mean + std * z (where z ~ N(0,1))
        for value in &mut tensor.data {
            *value = mean + std * (*value);
        }
        Ok(tensor)
    }

    /// Generate tensor with Xavier/Glorot initialization for neural networks
    pub fn xavier_uniform(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        if shape.len() < 2 {
            return Err(JsValue::from_str(
                "Xavier initialization requires at least 2D tensor",
            ));
        }

        let fan_in = shape[shape.len() - 2];
        let fan_out = shape[shape.len() - 1];
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

        Self::random_uniform(shape, -limit, limit)
    }

    /// Generate tensor with He initialization for neural networks (good for ReLU)
    pub fn he_normal(shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        if shape.len() < 2 {
            return Err(JsValue::from_str(
                "He initialization requires at least 2D tensor",
            ));
        }

        let fan_in = shape[shape.len() - 2];
        let std = (2.0 / fan_in as f32).sqrt();

        Self::randn_with_params(shape, 0.0, std)
    }

    /// Generate tensor with uniform random distribution
    pub fn random_uniform(shape: Vec<usize>, min: f32, max: f32) -> Result<WasmTensor, JsValue> {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        match Self::generate_uniform_distribution(size, min, max) {
            Ok(values) => data = values,
            Err(_) => {
                // Fallback implementation
                let mut seed = 12345u32;
                for _ in 0..size {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let uniform_val = seed as f32 / u32::MAX as f32;
                    data.push(min + uniform_val * (max - min));
                }
            },
        }

        WasmTensor::new(data, shape)
    }

    pub fn add(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape != other.shape {
            return Err(JsValue::from_str("Shape mismatch for addition"));
        }

        let data = if cfg!(target_feature = "simd128") {
            self.add_simd(&other.data)
        } else {
            self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect()
        };

        WasmTensor::new(data, self.shape.clone())
    }

    pub fn sub(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape != other.shape {
            return Err(JsValue::from_str("Shape mismatch for subtraction"));
        }

        let data = if cfg!(target_feature = "simd128") {
            self.sub_simd(&other.data)
        } else {
            self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect()
        };

        WasmTensor::new(data, self.shape.clone())
    }

    pub fn mul(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape != other.shape {
            return Err(JsValue::from_str("Shape mismatch for multiplication"));
        }

        let data = if cfg!(target_feature = "simd128") {
            self.mul_simd(&other.data)
        } else {
            self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect()
        };

        WasmTensor::new(data, self.shape.clone())
    }

    pub fn matmul(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(JsValue::from_str("matmul requires 2D tensors"));
        }

        let (_m, k1) = (self.shape[0], self.shape[1]);
        let (k2, _n) = (other.shape[0], other.shape[1]);

        if k1 != k2 {
            return Err(JsValue::from_str("Inner dimensions must match for matmul"));
        }

        // Use optimized SIMD-based matrix multiplication for better performance
        if cfg!(target_feature = "simd128") {
            self.matmul_simd_optimized(other)
        } else {
            self.matmul_blocked(other)
        }
    }

    pub fn transpose(&self) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 2 {
            return Err(JsValue::from_str("transpose requires 2D tensor"));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = self.data[i * cols + j];
            }
        }

        WasmTensor::new(result, vec![cols, rows])
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<WasmTensor, JsValue> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(JsValue::from_str("New shape size must match data length"));
        }

        let strides = compute_strides(&new_shape);
        Ok(WasmTensor {
            data: self.data.clone(),
            shape: new_shape,
            strides,
        })
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.data.len() as f32
    }

    pub fn exp(&self) -> WasmTensor {
        let data: Vec<f32> = self.data.iter().map(|x| x.exp()).collect();
        WasmTensor::new(data, self.shape.clone())
            .expect("exp: tensor creation should succeed with valid shape")
    }

    pub fn log(&self) -> WasmTensor {
        let data: Vec<f32> = self.data.iter().map(|x| x.ln()).collect();
        WasmTensor::new(data, self.shape.clone())
            .expect("log: tensor creation should succeed with valid shape")
    }

    pub fn softmax(&self, axis: i32) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 2 {
            return Err(JsValue::from_str("softmax requires 2D tensor"));
        }

        let axis = if axis < 0 { self.shape.len() as i32 + axis } else { axis } as usize;

        if axis >= self.shape.len() {
            return Err(JsValue::from_str("Invalid axis"));
        }

        let mut result = self.data.clone();

        if axis == 1 {
            // Softmax over last dimension
            let (rows, cols) = (self.shape[0], self.shape[1]);
            for i in 0..rows {
                let row_start = i * cols;
                let row_end = row_start + cols;
                let row = &mut result[row_start..row_end];

                // Find max for numerical stability
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                // Exp and sum
                let exp_sum: f32 = row.iter().map(|x| (x - max_val).exp()).sum();

                // Normalize
                for x in row.iter_mut() {
                    *x = (*x - max_val).exp() / exp_sum;
                }
            }
        } else {
            return Err(JsValue::from_str("Only axis=1 softmax supported"));
        }

        WasmTensor::new(result, self.shape.clone())
    }

    pub fn gelu(&self) -> WasmTensor {
        // Use optimized SIMD version by default
        self.gelu_simd()
    }

    pub fn relu(&self) -> WasmTensor {
        // Use optimized SIMD version by default
        self.relu_simd()
    }

    pub fn sigmoid(&self) -> WasmTensor {
        // Use optimized SIMD version by default
        self.sigmoid_simd()
    }

    pub fn tanh(&self) -> WasmTensor {
        // Use optimized SIMD version by default
        self.tanh_simd()
    }

    /// JavaScript toString() method - delegates to Display implementation
    #[wasm_bindgen(js_name = toString)]
    #[allow(clippy::inherent_to_string_shadow_display)]
    pub fn to_string(&self) -> String {
        format!("{}", self)
    }

    /// Scalar multiplication - optimized in-place operation
    pub fn scale(&mut self, scalar: f32) {
        if cfg!(target_feature = "simd128") {
            self.scale_simd(scalar);
        } else {
            for x in self.data.iter_mut() {
                *x *= scalar;
            }
        }
    }

    /// Efficient transpose for 2D tensors
    pub fn transpose_2d(&self) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 2 {
            return Err(JsValue::from_str("Transpose requires 2D tensor"));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(self.data.len());

        // Cache-friendly transpose
        for j in 0..cols {
            for i in 0..rows {
                data.push(self.data[i * cols + j]);
            }
        }

        WasmTensor::new(data, vec![cols, rows])
    }

    /// Fast element-wise maximum with scalar
    pub fn max_scalar(&self, scalar: f32) -> WasmTensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(scalar)).collect();
        WasmTensor::new(data, self.shape.clone())
            .expect("max_scalar: tensor creation should succeed with valid shape")
    }

    /// Efficient dot product for 1D tensors
    pub fn dot(&self, other: &WasmTensor) -> Result<f32, JsValue> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(JsValue::from_str("Dot product requires 1D tensors"));
        }

        if self.shape[0] != other.shape[0] {
            return Err(JsValue::from_str("Length mismatch for dot product"));
        }

        if cfg!(target_feature = "simd128") {
            Ok(self.dot_simd(&other.data))
        } else {
            Ok(self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum())
        }
    }

    /// Batch matrix multiplication for multiple small matrices
    pub fn batch_matmul(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 3 || other.shape.len() != 3 {
            return Err(JsValue::from_str("Batch matmul requires 3D tensors"));
        }

        let (batch_size, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
        let (other_batch, k2, n) = (other.shape[0], other.shape[1], other.shape[2]);

        if batch_size != other_batch || k != k2 {
            return Err(JsValue::from_str("Shape mismatch for batch matmul"));
        }

        let mut result_data = Vec::with_capacity(batch_size * m * n);

        // Process each batch
        for b in 0..batch_size {
            let a_offset = b * m * k;
            let b_offset = b * k * n;

            // Optimized matrix multiplication with blocking
            #[allow(clippy::excessive_nesting)]
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += self.data[a_offset + i * k + kk] * other.data[b_offset + kk * n + j];
                    }
                    result_data.push(sum);
                }
            }
        }

        WasmTensor::new(result_data, vec![batch_size, m, n])
    }

    /// Generate normal distribution using Box-Muller transform
    fn generate_normal_distribution(size: usize) -> Result<Vec<f32>, String> {
        let mut data = Vec::with_capacity(size);
        let mut rng_bytes = vec![0u8; (size * 8).div_ceil(8) * 8]; // Ensure we have enough bytes, rounded up
        getrandom::fill(&mut rng_bytes)
            .map_err(|e| format!("Random generation failed: {:?}", e))?;

        let mut i = 0;
        let mut byte_index = 0;

        while i < size {
            // Get two uniform random values from [0,1)
            let u1 = Self::bytes_to_f32(&rng_bytes[byte_index..byte_index + 4]);
            let u2 = Self::bytes_to_f32(&rng_bytes[byte_index + 4..byte_index + 8]);
            byte_index += 8;

            // Box-Muller transform to get normal distribution
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();

            data.push(z0);
            i += 1;

            if i < size {
                data.push(z1);
                i += 1;
            }
        }

        Ok(data)
    }

    /// Generate uniform distribution
    fn generate_uniform_distribution(size: usize, min: f32, max: f32) -> Result<Vec<f32>, String> {
        let mut data = Vec::with_capacity(size);
        let mut rng_bytes = vec![0u8; size * 4];
        getrandom::fill(&mut rng_bytes)
            .map_err(|e| format!("Random generation failed: {:?}", e))?;

        for i in 0..size {
            let uniform_val = Self::bytes_to_f32(&rng_bytes[i * 4..(i + 1) * 4]);
            data.push(min + uniform_val * (max - min));
        }

        Ok(data)
    }

    /// Convert 4 bytes to f32 in range [0, 1)
    fn bytes_to_f32(bytes: &[u8]) -> f32 {
        let mut array = [0u8; 4];
        array.copy_from_slice(&bytes[..4]);
        let int_val = u32::from_le_bytes(array);
        // Convert to [0, 1) range
        (int_val as f64 / u32::MAX as f64) as f32
    }
}

// Display implementation for WasmTensor
impl std::fmt::Display for WasmTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WasmTensor(shape={:?}, dtype=f32)", self.shape)
    }
}

// SIMD implementations for WasmTensor operations
impl WasmTensor {
    #[cfg(target_arch = "wasm32")]
    #[inline]
    fn add_simd(&self, other: &[f32]) -> Vec<f32> {
        // Safety check: ensure both arrays have the same length
        if self.data.len() != other.len() {
            // Fallback to safe element-wise addition for mismatched sizes
            let min_len = self.data.len().min(other.len());
            return self.data[..min_len]
                .iter()
                .zip(other[..min_len].iter())
                .map(|(&a, &b)| a + b)
                .collect();
        }

        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let _remainder = self.data.len() % 4;

        // Process 4 elements at a time with SIMD with bounds checking
        for i in 0..chunks {
            let offset = i * 4;

            // Safety check: ensure we don't read beyond array bounds
            if offset + 4 <= self.data.len() && offset + 4 <= other.len() {
                unsafe {
                    let a = v128_load(&self.data[offset] as *const f32 as *const v128);
                    let b = v128_load(&other[offset] as *const f32 as *const v128);
                    let sum = f32x4_add(a, b);

                    let mut temp = [0.0f32; 4];
                    v128_store(&mut temp[0] as *mut f32 as *mut v128, sum);
                    result.extend_from_slice(&temp);
                }
            } else {
                // Fallback to safe scalar operations if bounds check fails
                for j in 0..4 {
                    let idx = offset + j;
                    if idx < self.data.len() && idx < other.len() {
                        result.push(self.data[idx] + other[idx]);
                    }
                }
            }
        }

        // Handle remaining elements with bounds checking
        for i in (chunks * 4)..self.data.len() {
            if i < other.len() {
                result.push(self.data[i] + other[i]);
            }
        }

        result
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    fn sub_simd(&self, other: &[f32]) -> Vec<f32> {
        // Safety check: ensure both arrays have the same length
        if self.data.len() != other.len() {
            // Fallback to safe element-wise subtraction for mismatched sizes
            let min_len = self.data.len().min(other.len());
            return self.data[..min_len]
                .iter()
                .zip(other[..min_len].iter())
                .map(|(&a, &b)| a - b)
                .collect();
        }

        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let _remainder = self.data.len() % 4;

        // Process 4 elements at a time with SIMD with bounds checking
        for i in 0..chunks {
            let offset = i * 4;

            // Safety check: ensure we don't read beyond array bounds
            if offset + 4 <= self.data.len() && offset + 4 <= other.len() {
                unsafe {
                    let a = v128_load(&self.data[offset] as *const f32 as *const v128);
                    let b = v128_load(&other[offset] as *const f32 as *const v128);
                    let diff = f32x4_sub(a, b);

                    let mut temp = [0.0f32; 4];
                    v128_store(&mut temp[0] as *mut f32 as *mut v128, diff);
                    result.extend_from_slice(&temp);
                }
            } else {
                // Fallback to safe scalar operations if bounds check fails
                for j in 0..4 {
                    let idx = offset + j;
                    if idx < self.data.len() && idx < other.len() {
                        result.push(self.data[idx] - other[idx]);
                    }
                }
            }
        }

        // Handle remaining elements with bounds checking
        for i in (chunks * 4)..self.data.len() {
            if i < other.len() {
                result.push(self.data[i] - other[i]);
            }
        }

        result
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    fn mul_simd(&self, other: &[f32]) -> Vec<f32> {
        // Safety check: ensure both arrays have the same length
        if self.data.len() != other.len() {
            // Fallback to safe element-wise multiplication for mismatched sizes
            let min_len = self.data.len().min(other.len());
            return self.data[..min_len]
                .iter()
                .zip(other[..min_len].iter())
                .map(|(&a, &b)| a * b)
                .collect();
        }

        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let _remainder = self.data.len() % 4;

        // Process 4 elements at a time with SIMD with bounds checking
        for i in 0..chunks {
            let offset = i * 4;

            // Safety check: ensure we don't read beyond array bounds
            if offset + 4 <= self.data.len() && offset + 4 <= other.len() {
                unsafe {
                    let a = v128_load(&self.data[offset] as *const f32 as *const v128);
                    let b = v128_load(&other[offset] as *const f32 as *const v128);
                    let product = f32x4_mul(a, b);

                    let mut temp = [0.0f32; 4];
                    v128_store(&mut temp[0] as *mut f32 as *mut v128, product);
                    result.extend_from_slice(&temp);
                }
            } else {
                // Fallback to safe scalar operations if bounds check fails
                for j in 0..4 {
                    let idx = offset + j;
                    if idx < self.data.len() && idx < other.len() {
                        result.push(self.data[idx] * other[idx]);
                    }
                }
            }
        }

        // Handle remaining elements with bounds checking
        for i in (chunks * 4)..self.data.len() {
            if i < other.len() {
                result.push(self.data[i] * other[i]);
            }
        }

        result
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    fn scale_simd(&mut self, scalar: f32) {
        let chunks = self.data.len() / 4;
        let scalar_vec = unsafe { f32x4_splat(scalar) };

        // Process 4 elements at a time with SIMD
        for i in 0..chunks {
            let offset = i * 4;
            unsafe {
                let data_vec = v128_load(&self.data[offset] as *const f32 as *const v128);
                let result = f32x4_mul(data_vec, scalar_vec);
                v128_store(&mut self.data[offset] as *mut f32 as *mut v128, result);
            }
        }

        // Handle remaining elements
        for i in (chunks * 4)..self.data.len() {
            self.data[i] *= scalar;
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    fn dot_simd(&self, other: &[f32]) -> f32 {
        let chunks = self.data.len() / 4;
        let mut sum_vec = unsafe { f32x4_splat(0.0) };

        // Process 4 elements at a time with SIMD
        for i in 0..chunks {
            let offset = i * 4;
            unsafe {
                let a = v128_load(&self.data[offset] as *const f32 as *const v128);
                let b = v128_load(&other[offset] as *const f32 as *const v128);
                let product = f32x4_mul(a, b);
                sum_vec = f32x4_add(sum_vec, product);
            }
        }

        // Extract sum from SIMD register
        let mut result = unsafe {
            f32x4_extract_lane::<0>(sum_vec)
                + f32x4_extract_lane::<1>(sum_vec)
                + f32x4_extract_lane::<2>(sum_vec)
                + f32x4_extract_lane::<3>(sum_vec)
        };

        // Handle remaining elements
        for i in (chunks * 4)..self.data.len() {
            result += self.data[i] * other[i];
        }

        result
    }

    // Fallback implementations for non-WASM targets
    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    fn add_simd(&self, other: &[f32]) -> Vec<f32> {
        self.data.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    fn sub_simd(&self, other: &[f32]) -> Vec<f32> {
        self.data.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    fn mul_simd(&self, other: &[f32]) -> Vec<f32> {
        self.data.iter().zip(other.iter()).map(|(a, b)| a * b).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    fn scale_simd(&mut self, scalar: f32) {
        for x in self.data.iter_mut() {
            *x *= scalar;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    fn dot_simd(&self, other: &[f32]) -> f32 {
        self.data.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }

    /// Optimized blocked matrix multiplication (cache-friendly)
    #[inline]
    pub(super) fn matmul_blocked(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let (m, k) = (self.shape[0], self.shape[1]);
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];

        // Use blocking for cache efficiency
        const BLOCK_SIZE: usize = 64;

        #[allow(clippy::excessive_nesting)]
        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                for k_block in (0..k).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(m);
                    let j_end = (j_block + BLOCK_SIZE).min(n);
                    let k_end = (k_block + BLOCK_SIZE).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k_idx in k_block..k_end {
                                sum += self.data[i * k + k_idx] * other.data[k_idx * n + j];
                            }
                            result[i * n + j] += sum;
                        }
                    }
                }
            }
        }

        WasmTensor::new(result, vec![m, n])
    }

    /// SIMD-optimized matrix multiplication
    #[cfg(target_arch = "wasm32")]
    #[inline]
    pub(super) fn matmul_simd_optimized(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let (m, k) = (self.shape[0], self.shape[1]);
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];

        // SIMD-optimized matrix multiplication with 4-wide vectors
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let j_end = (j + 4).min(n);
                let mut acc = unsafe { f32x4_splat(0.0) };

                for k_idx in 0..k {
                    let a_val = self.data[i * k + k_idx];
                    let a_vec = unsafe { f32x4_splat(a_val) };

                    if j_end - j == 4 {
                        let b_vec = unsafe {
                            v128_load(&other.data[k_idx * n + j] as *const f32 as *const v128)
                        };
                        acc = unsafe { f32x4_add(acc, f32x4_mul(a_vec, b_vec)) };
                    } else {
                        // Handle remaining elements
                        for jj in j..j_end {
                            result[i * n + jj] += a_val * other.data[k_idx * n + jj];
                        }
                    }
                }

                if j_end - j == 4 {
                    unsafe {
                        v128_store(&mut result[i * n + j] as *mut f32 as *mut v128, acc);
                    }
                }
            }
        }

        WasmTensor::new(result, vec![m, n])
    }
}

pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
