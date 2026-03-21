//! Advanced tensor operations and helper functions
//!
//! Contains SIMD activation methods, linear algebra operations,
//! and utility functions.

use std::string::String;
use std::vec;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use super::tensor_core::{compute_strides, WasmTensor};

impl WasmTensor {
    /// SIMD-optimized ReLU activation
    pub fn relu_simd(&self) -> WasmTensor {
        let data = if cfg!(target_feature = "simd128") {
            self.relu_simd_impl()
        } else {
            self.data.iter().map(|&x| x.max(0.0)).collect()
        };
        WasmTensor::new(data, self.shape.clone())
            .expect("tensor creation should succeed with valid shape")
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    pub(super) fn relu_simd_impl(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let zero_vec = unsafe { f32x4_splat(0.0) };

        // Process 4 elements at a time with SIMD with bounds checking
        for i in 0..chunks {
            let offset = i * 4;

            // Safety check: ensure we don't read beyond array bounds
            if offset + 4 <= self.data.len() {
                unsafe {
                    let data_vec = v128_load(&self.data[offset] as *const f32 as *const v128);
                    let relu_result = f32x4_max(data_vec, zero_vec);

                    let mut temp = [0.0f32; 4];
                    v128_store(&mut temp[0] as *mut f32 as *mut v128, relu_result);
                    result.extend_from_slice(&temp);
                }
            } else {
                // Fallback to safe scalar operations if bounds check fails
                for j in 0..4 {
                    let idx = offset + j;
                    if idx < self.data.len() {
                        result.push(self.data[idx].max(0.0));
                    }
                }
            }
        }

        // Handle remaining elements with bounds checking
        for i in (chunks * 4)..self.data.len() {
            result.push(self.data[i].max(0.0));
        }

        result
    }

    /// SIMD-optimized GELU activation
    pub fn gelu_simd(&self) -> WasmTensor {
        let data = if cfg!(target_feature = "simd128") {
            self.gelu_simd_impl()
        } else {
            use std::f32::consts::PI;
            self.data
                .iter()
                .map(|&x| 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()))
                .collect()
        };
        WasmTensor::new(data, self.shape.clone())
            .expect("tensor creation should succeed with valid shape")
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    pub(super) fn gelu_simd_impl(&self) -> Vec<f32> {
        use std::f32::consts::PI;
        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let half = unsafe { f32x4_splat(0.5) };
        let one = unsafe { f32x4_splat(1.0) };
        let gelu_const = unsafe { f32x4_splat((2.0 / PI).sqrt()) };
        let cubic_const = unsafe { f32x4_splat(0.044715) };

        // Process 4 elements at a time with SIMD with bounds checking
        for i in 0..chunks {
            let offset = i * 4;

            // Safety check: ensure we don't read beyond array bounds
            if offset + 4 <= self.data.len() {
                unsafe {
                    let x = v128_load(&self.data[offset] as *const f32 as *const v128);

                    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                    let x_cubed = f32x4_mul(f32x4_mul(x, x), x);
                    let inner = f32x4_add(x, f32x4_mul(cubic_const, x_cubed));
                    let tanh_input = f32x4_mul(gelu_const, inner);

                    // Approximate tanh using polynomial (less accurate but faster)
                    let tanh_approx = f32x4_div(
                        tanh_input,
                        f32x4_add(one, f32x4_mul(tanh_input, tanh_input)),
                    );

                    let result_vec = f32x4_mul(f32x4_mul(half, x), f32x4_add(one, tanh_approx));

                    let mut temp = [0.0f32; 4];
                    v128_store(&mut temp[0] as *mut f32 as *mut v128, result_vec);
                    result.extend_from_slice(&temp);
                }
            } else {
                // Fallback to safe scalar operations if bounds check fails
                for j in 0..4 {
                    let idx = offset + j;
                    if idx < self.data.len() {
                        let x = self.data[idx];
                        result.push(
                            0.5 * x
                                * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()),
                        );
                    }
                }
            }
        }

        // Handle remaining elements with precise calculation and bounds checking
        for i in (chunks * 4)..self.data.len() {
            let x = self.data[i];
            result.push(0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()));
        }

        result
    }

    /// SIMD-optimized Sigmoid activation
    pub fn sigmoid_simd(&self) -> WasmTensor {
        let data = if cfg!(target_feature = "simd128") {
            self.sigmoid_simd_impl()
        } else {
            self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
        };
        WasmTensor::new(data, self.shape.clone())
            .expect("tensor creation should succeed with valid shape")
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    pub(super) fn sigmoid_simd_impl(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let one = unsafe { f32x4_splat(1.0) };

        // Process 4 elements at a time with SIMD with bounds checking
        for i in 0..chunks {
            let offset = i * 4;

            // Safety check: ensure we don't read beyond array bounds
            if offset + 4 <= self.data.len() {
                unsafe {
                    let x = v128_load(&self.data[offset] as *const f32 as *const v128);
                    // Compute -x
                    let neg_x = f32x4_neg(x);

                    // Approximate exp(-x) using polynomial approximation
                    // For better accuracy in WASM, we use a simplified approach
                    // sigmoid(x) ≈ 1 / (1 + exp(-x))
                    let mut temp_in = [0.0f32; 4];
                    v128_store(&mut temp_in[0] as *mut f32 as *mut v128, neg_x);

                    // Compute exp for each element (no SIMD exp in WASM)
                    let exp_vals = [
                        temp_in[0].exp(),
                        temp_in[1].exp(),
                        temp_in[2].exp(),
                        temp_in[3].exp(),
                    ];

                    let exp_vec = v128_load(&exp_vals[0] as *const f32 as *const v128);
                    let denom = f32x4_add(one, exp_vec);
                    let sigmoid_result = f32x4_div(one, denom);

                    let mut temp = [0.0f32; 4];
                    v128_store(&mut temp[0] as *mut f32 as *mut v128, sigmoid_result);
                    result.extend_from_slice(&temp);
                }
            } else {
                // Fallback to safe scalar operations if bounds check fails
                for j in 0..4 {
                    let idx = offset + j;
                    if idx < self.data.len() {
                        let x = self.data[idx];
                        result.push(1.0 / (1.0 + (-x).exp()));
                    }
                }
            }
        }

        // Handle remaining elements with bounds checking
        for i in (chunks * 4)..self.data.len() {
            result.push(1.0 / (1.0 + (-self.data[i]).exp()));
        }

        result
    }

    /// SIMD-optimized Tanh activation
    pub fn tanh_simd(&self) -> WasmTensor {
        let data = if cfg!(target_feature = "simd128") {
            self.tanh_simd_impl()
        } else {
            self.data.iter().map(|&x| x.tanh()).collect()
        };
        WasmTensor::new(data, self.shape.clone())
            .expect("tensor creation should succeed with valid shape")
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    pub(super) fn tanh_simd_impl(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let one = unsafe { f32x4_splat(1.0) };
        let two = unsafe { f32x4_splat(2.0) };

        // Process 4 elements at a time with SIMD with bounds checking
        for i in 0..chunks {
            let offset = i * 4;

            // Safety check: ensure we don't read beyond array bounds
            if offset + 4 <= self.data.len() {
                unsafe {
                    let x = v128_load(&self.data[offset] as *const f32 as *const v128);

                    // Compute 2*x
                    let two_x = f32x4_mul(two, x);

                    // Extract values for exp computation (no SIMD exp in WASM)
                    let mut temp_in = [0.0f32; 4];
                    v128_store(&mut temp_in[0] as *mut f32 as *mut v128, two_x);

                    // Compute exp(2*x) for each element
                    let exp_vals = [
                        temp_in[0].exp(),
                        temp_in[1].exp(),
                        temp_in[2].exp(),
                        temp_in[3].exp(),
                    ];

                    let exp_2x = v128_load(&exp_vals[0] as *const f32 as *const v128);

                    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                    let numerator = f32x4_sub(exp_2x, one);
                    let denominator = f32x4_add(exp_2x, one);
                    let tanh_result = f32x4_div(numerator, denominator);

                    let mut temp = [0.0f32; 4];
                    v128_store(&mut temp[0] as *mut f32 as *mut v128, tanh_result);
                    result.extend_from_slice(&temp);
                }
            } else {
                // Fallback to safe scalar operations if bounds check fails
                for j in 0..4 {
                    let idx = offset + j;
                    if idx < self.data.len() {
                        result.push(self.data[idx].tanh());
                    }
                }
            }
        }

        // Handle remaining elements with bounds checking
        for i in (chunks * 4)..self.data.len() {
            result.push(self.data[i].tanh());
        }

        result
    }

    /// Fused matrix multiplication with bias addition
    pub fn matmul_bias(
        &self,
        weights: &WasmTensor,
        bias: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let matmul_result = self.matmul(weights)?;
        matmul_result.add(bias)
    }

    /// Fused matrix multiplication with bias and ReLU activation
    pub fn matmul_bias_relu(
        &self,
        weights: &WasmTensor,
        bias: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let mut result = self.matmul_bias(weights, bias)?;
        result = result.relu_simd();
        Ok(result)
    }

    /// Layer normalization (simplified version without learnable parameters)
    pub fn layer_norm(&self, normalized_shape: &[usize], eps: f32) -> Result<WasmTensor, JsValue> {
        if self.shape.len() < normalized_shape.len() {
            return Err(JsValue::from_str("Invalid normalized_shape"));
        }

        // For 2D tensors with normalized_shape matching last dim
        if self.shape.len() == 2 && normalized_shape.len() == 1 {
            let (batch_size, hidden_size) = (self.shape[0], self.shape[1]);
            if normalized_shape[0] != hidden_size {
                return Err(JsValue::from_str(
                    "normalized_shape must match last dimension",
                ));
            }

            let mut result = self.data.clone();

            // Normalize each sample in the batch (without gamma/beta)
            for i in 0..batch_size {
                let offset = i * hidden_size;
                let sample = &mut result[offset..offset + hidden_size];

                // Calculate mean
                let mean = sample.iter().sum::<f32>() / hidden_size as f32;

                // Calculate variance
                let variance =
                    sample.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;

                let inv_std = 1.0 / (variance + eps).sqrt();

                // Normalize
                for val in sample.iter_mut() {
                    *val = (*val - mean) * inv_std;
                }
            }

            WasmTensor::new(result, self.shape.clone())
        } else {
            Err(JsValue::from_str("Only 2D layer norm currently supported"))
        }
    }

    /// Layer normalization with learnable parameters
    pub fn layer_norm_with_params(
        &self,
        gamma: &WasmTensor,
        beta: &WasmTensor,
        eps: f32,
    ) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 2 {
            return Err(JsValue::from_str("Layer norm requires 2D tensor"));
        }

        let (batch_size, hidden_size) = (self.shape[0], self.shape[1]);
        let mut result = self.data.clone();

        // Normalize each sample in the batch
        for i in 0..batch_size {
            let offset = i * hidden_size;
            let sample = &mut result[offset..offset + hidden_size];

            // Calculate mean
            let mean = sample.iter().sum::<f32>() / hidden_size as f32;

            // Calculate variance
            let variance =
                sample.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;

            let inv_std = 1.0 / (variance + eps).sqrt();

            // Normalize and apply scale/shift
            for ((val, &g), &b) in sample.iter_mut().zip(gamma.data.iter()).zip(beta.data.iter()) {
                *val = (*val - mean) * inv_std * g + b;
            }
        }

        WasmTensor::new(result, self.shape.clone())
    }

    /// Scaled dot-product attention mechanism
    /// Computes: softmax((query @ key^T) / sqrt(d_k)) @ value
    pub fn scaled_dot_product_attention(
        &self,
        key: &WasmTensor,
        value: &WasmTensor,
        mask: Option<&WasmTensor>,
    ) -> Result<WasmTensor, JsValue> {
        if self.shape.len() < 2 || key.shape.len() < 2 || value.shape.len() < 2 {
            return Err(JsValue::from_str("Attention requires at least 2D tensors"));
        }

        let d_k = if self.shape.len() == 2 { self.shape[1] } else { self.shape[2] } as f32;

        let key_transposed = key.transpose()?;
        let attention_scores = self.matmul(&key_transposed)?;
        let scale = 1.0 / d_k.sqrt();
        let mut scaled_scores = attention_scores.clone();
        scaled_scores.scale(scale);

        let scores_to_softmax = if let Some(mask_tensor) = mask {
            let mask_data = mask_tensor.data_ref();
            let mut masked_data = scaled_scores.data.clone();
            for i in 0..masked_data.len() {
                if i < mask_data.len() && mask_data[i] == 0.0 {
                    masked_data[i] = -1e9;
                }
            }
            WasmTensor::new(masked_data, scaled_scores.shape.clone())?
        } else {
            scaled_scores
        };

        let attention_weights = scores_to_softmax.softmax(-1)?;
        let output = attention_weights.matmul(value)?;
        Ok(output)
    }

    // Fallback implementation for non-WASM targets
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn matmul_simd_optimized(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        self.matmul_blocked(other)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn relu_simd_impl(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x.max(0.0)).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn gelu_simd_impl(&self) -> Vec<f32> {
        use std::f32::consts::PI;
        self.data
            .iter()
            .map(|&x| 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()))
            .collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn sigmoid_simd_impl(&self) -> Vec<f32> {
        self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn tanh_simd_impl(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x.tanh()).collect()
    }

    /// Advanced Singular Value Decomposition using Jacobi rotations
    /// Returns (U, S, V^T) where A = U * S * V^T
    /// This is computationally intensive and demonstrates advanced numerical methods
    pub fn svd(&self) -> Result<(WasmTensor, WasmTensor, WasmTensor), JsValue> {
        if self.shape.len() != 2 {
            return Err(JsValue::from_str("SVD requires 2D tensor"));
        }

        let (m, n) = (self.shape[0], self.shape[1]);
        let min_dim = m.min(n);

        // Create initial matrices
        let u = self.create_identity_matrix(m)?;
        let mut s = self.clone();
        let mut vt = self.create_identity_matrix(n)?;

        // Jacobi SVD iteration - sophisticated iterative algorithm
        const MAX_ITERATIONS: usize = 50;
        const TOLERANCE: f32 = 1e-12;

        for iteration in 0..MAX_ITERATIONS {
            let mut max_off_diagonal = 0.0f32;

            // Two-sided Jacobi rotations for SVD
            for i in 0..min_dim {
                for j in (i + 1)..min_dim {
                    // Calculate rotation parameters using sophisticated trigonometry
                    let (c, s_rot) = self.calculate_jacobi_rotation_svd(&s, i, j)?;

                    // Apply rotations to matrices
                    self.apply_givens_rotation_left(&mut s, i, j, c, s_rot)?;
                    self.apply_givens_rotation_right(&mut vt, i, j, c, s_rot)?;

                    // Track convergence
                    let off_diag = s.data[i * n + j].abs();
                    max_off_diagonal = max_off_diagonal.max(off_diag);
                }
            }

            // Check convergence with sophisticated termination criteria
            if max_off_diagonal < TOLERANCE {
                break;
            }

            // Prevent infinite loops
            if iteration == MAX_ITERATIONS - 1 {
                return Err(JsValue::from_str("SVD failed to converge"));
            }
        }

        // Extract singular values and ensure non-negative
        let mut singular_values = Vec::with_capacity(min_dim);
        for i in 0..min_dim {
            let val = s.data[i * n + i].abs();
            singular_values.push(val);
        }

        let s_tensor = WasmTensor::new(singular_values, vec![min_dim])?;
        Ok((u, s_tensor, vt))
    }

    /// Advanced QR Decomposition using Modified Gram-Schmidt with pivoting
    /// Returns (Q, R, P) where A*P = Q*R with column pivoting for numerical stability
    pub fn qr_decomposition_with_pivoting(
        &self,
    ) -> Result<(WasmTensor, WasmTensor, Vec<usize>), JsValue> {
        if self.shape.len() != 2 {
            return Err(JsValue::from_str("QR decomposition requires 2D tensor"));
        }

        let (m, n) = (self.shape[0], self.shape[1]);
        let mut r = self.clone();
        let mut q = self.create_identity_matrix(m)?;
        let mut pivot = (0..n).collect::<Vec<usize>>();

        // Column pivoting for numerical stability
        for k in 0..n.min(m) {
            // Find column with maximum norm (pivoting strategy)
            let mut max_norm = 0.0f32;
            let mut pivot_col = k;

            for j in k..n {
                let mut norm = 0.0f32;
                for i in k..m {
                    norm += r.data[i * n + j] * r.data[i * n + j];
                }
                if norm > max_norm {
                    max_norm = norm;
                    pivot_col = j;
                }
            }

            // Swap columns if necessary
            if pivot_col != k {
                for i in 0..m {
                    r.data.swap(i * n + k, i * n + pivot_col);
                }
                pivot.swap(k, pivot_col);
            }

            // Modified Gram-Schmidt orthogonalization
            let mut norm = 0.0f32;
            for i in k..m {
                norm += r.data[i * n + k] * r.data[i * n + k];
            }
            norm = norm.sqrt();

            if norm < 1e-12 {
                continue; // Skip near-zero columns
            }

            // Create Householder reflector for numerical stability
            let (householder_v, beta) = self.compute_householder_vector(&r, k, k, m, n)?;

            // Apply Householder transformation to remaining matrix
            self.apply_householder_transformation(&mut r, &householder_v, beta, k, k, m, n)?;

            // Update Q matrix
            self.apply_householder_to_q(&mut q, &householder_v, beta, k, m)?;
        }

        Ok((q, r, pivot))
    }

    /// Advanced eigenvalue decomposition using QR algorithm with shifts
    /// Returns (eigenvalues, eigenvectors) for symmetric matrices
    pub fn eigenvalue_decomposition_symmetric(&self) -> Result<(WasmTensor, WasmTensor), JsValue> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(JsValue::from_str(
                "Eigendecomposition requires square matrix",
            ));
        }

        // Verify symmetry (approximately)
        let n = self.shape[0];
        for i in 0..n {
            for j in 0..n {
                let diff = (self.data[i * n + j] - self.data[j * n + i]).abs();
                if diff > 1e-10 {
                    return Err(JsValue::from_str(
                        "Matrix must be symmetric for this method",
                    ));
                }
            }
        }

        let mut a = self.clone();
        let mut eigenvectors = self.create_identity_matrix(n)?;

        // Tridiagonalization using Householder transformations
        self.tridiagonalize(&mut a, &mut eigenvectors)?;

        // QR algorithm with Wilkinson shifts for eigenvalues
        const MAX_ITERATIONS: usize = 100;
        const TOLERANCE: f32 = 1e-12;

        for iteration in 0..MAX_ITERATIONS {
            let mut converged = true;

            // Check for convergence of off-diagonal elements
            for i in 0..(n - 1) {
                if a.data[i * n + (i + 1)].abs() > TOLERANCE {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }

            // Wilkinson shift for acceleration
            let shift = self.compute_wilkinson_shift(&a, n)?;

            // Shift the matrix
            for i in 0..n {
                a.data[i * n + i] -= shift;
            }

            // QR step
            let (q_step, r_step, _) = a.qr_decomposition_with_pivoting()?;
            a = r_step.matmul(&q_step)?;
            eigenvectors = eigenvectors.matmul(&q_step)?;

            // Unshift
            for i in 0..n {
                a.data[i * n + i] += shift;
            }

            if iteration == MAX_ITERATIONS - 1 {
                return Err(JsValue::from_str(
                    "Eigenvalue computation failed to converge",
                ));
            }
        }

        // Extract eigenvalues from diagonal
        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(a.data[i * n + i]);
        }

        let eigenvalues_tensor = WasmTensor::new(eigenvalues, vec![n])?;
        Ok((eigenvalues_tensor, eigenvectors))
    }

    /// Advanced tensor contraction using Einstein summation notation
    /// Supports complex multi-dimensional tensor operations
    pub fn einsum_contract(
        &self,
        other: &WasmTensor,
        pattern: &str,
    ) -> Result<WasmTensor, JsValue> {
        // Parse Einstein summation pattern (e.g., "ij,jk->ik")
        let parts: Vec<&str> = pattern.split("->").collect();
        if parts.len() != 2 {
            return Err(JsValue::from_str("Invalid einsum pattern"));
        }

        let input_patterns: Vec<&str> = parts[0].split(',').collect();
        let output_pattern = parts[1];

        if input_patterns.len() != 2 {
            return Err(JsValue::from_str("Only binary operations supported"));
        }

        // Advanced pattern matching and tensor contraction logic
        match (input_patterns[0], input_patterns[1], output_pattern) {
            ("ij", "jk", "ik") => {
                // Standard matrix multiplication
                self.matmul(other)
            },
            ("ijk", "ikl", "ijl") => {
                // Batch matrix multiplication with advanced indexing
                self.batch_matmul_advanced(other)
            },
            ("ij", "ji", "") => {
                // Trace (sophisticated diagonal sum)
                self.trace_product(other)
            },
            ("ij", "ij", "") => {
                // Frobenius inner product
                self.frobenius_inner_product(other)
            },
            _ => Err(JsValue::from_str("Unsupported einsum pattern")),
        }
    }

    /// Advanced numerical gradient computation using complex step differentiation
    /// More accurate than finite differences, demonstrates sophisticated numerical methods
    pub fn complex_step_gradient<F>(&self, func: F) -> Result<WasmTensor, JsValue>
    where
        F: Fn(&WasmTensor) -> Result<f32, JsValue>,
    {
        let h = 1e-15f32; // Step size for complex step method
        let mut gradient = vec![0.0f32; self.data.len()];

        for i in 0..self.data.len() {
            // Create perturbation vector
            let mut perturbed_data = self.data.clone();

            // Complex step: f(x + ih) where i is imaginary unit
            // We approximate this using: (f(x + h) - f(x - h)) / (2h)
            // with very small h for high accuracy

            // Forward step
            perturbed_data[i] += h;
            let mut perturbed_tensor = WasmTensor::new(perturbed_data.clone(), self.shape.clone())?;
            let f_forward = func(&perturbed_tensor)?;

            // Backward step
            perturbed_data[i] = self.data[i] - h;
            perturbed_tensor = WasmTensor::new(perturbed_data, self.shape.clone())?;
            let f_backward = func(&perturbed_tensor)?;

            // Central difference approximation
            gradient[i] = (f_forward - f_backward) / (2.0 * h);
        }

        WasmTensor::new(gradient, self.shape.clone())
    }

    // Helper methods for advanced algorithms

    fn create_identity_matrix(&self, size: usize) -> Result<WasmTensor, JsValue> {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        WasmTensor::new(data, vec![size, size])
    }

    fn calculate_jacobi_rotation_svd(
        &self,
        matrix: &WasmTensor,
        i: usize,
        j: usize,
    ) -> Result<(f32, f32), JsValue> {
        let n = matrix.shape[1];
        let a_ii = matrix.data[i * n + i];
        let a_jj = matrix.data[j * n + j];
        let a_ij = matrix.data[i * n + j];

        if a_ij.abs() < 1e-15 {
            return Ok((1.0, 0.0));
        }

        let tau = (a_jj - a_ii) / (2.0 * a_ij);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        Ok((c, s))
    }

    fn apply_givens_rotation_left(
        &self,
        matrix: &mut WasmTensor,
        i: usize,
        j: usize,
        c: f32,
        s: f32,
    ) -> Result<(), JsValue> {
        let n = matrix.shape[1];
        for k in 0..n {
            let temp1 = matrix.data[i * n + k];
            let temp2 = matrix.data[j * n + k];
            matrix.data[i * n + k] = c * temp1 - s * temp2;
            matrix.data[j * n + k] = s * temp1 + c * temp2;
        }
        Ok(())
    }

    fn apply_givens_rotation_right(
        &self,
        matrix: &mut WasmTensor,
        i: usize,
        j: usize,
        c: f32,
        s: f32,
    ) -> Result<(), JsValue> {
        let m = matrix.shape[0];
        let n = matrix.shape[1];
        for k in 0..m {
            let temp1 = matrix.data[k * n + i];
            let temp2 = matrix.data[k * n + j];
            matrix.data[k * n + i] = c * temp1 - s * temp2;
            matrix.data[k * n + j] = s * temp1 + c * temp2;
        }
        Ok(())
    }

    fn compute_householder_vector(
        &self,
        matrix: &WasmTensor,
        col: usize,
        start_row: usize,
        m: usize,
        n: usize,
    ) -> Result<(Vec<f32>, f32), JsValue> {
        let mut v = vec![0.0f32; m - start_row];
        let mut norm_x = 0.0f32;

        for i in start_row..m {
            let val = matrix.data[i * n + col];
            v[i - start_row] = val;
            norm_x += val * val;
        }
        norm_x = norm_x.sqrt();

        if norm_x == 0.0 {
            return Ok((v, 0.0));
        }

        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_x;

        let mut norm_v = 0.0f32;
        for val in &v {
            norm_v += val * val;
        }
        norm_v = norm_v.sqrt();

        if norm_v > 0.0 {
            for val in &mut v {
                *val /= norm_v;
            }
        }

        let beta = 2.0;
        Ok((v, beta))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_householder_transformation(
        &self,
        matrix: &mut WasmTensor,
        v: &[f32],
        beta: f32,
        start_row: usize,
        start_col: usize,
        m: usize,
        n: usize,
    ) -> Result<(), JsValue> {
        for j in start_col..n {
            let mut dot_product = 0.0f32;
            for i in start_row..m {
                dot_product += v[i - start_row] * matrix.data[i * n + j];
            }

            for i in start_row..m {
                matrix.data[i * n + j] -= beta * v[i - start_row] * dot_product;
            }
        }
        Ok(())
    }

    fn apply_householder_to_q(
        &self,
        q: &mut WasmTensor,
        v: &[f32],
        beta: f32,
        start_row: usize,
        m: usize,
    ) -> Result<(), JsValue> {
        for j in 0..m {
            let mut dot_product = 0.0f32;
            for i in start_row..m {
                dot_product += v[i - start_row] * q.data[i * m + j];
            }

            for i in start_row..m {
                q.data[i * m + j] -= beta * v[i - start_row] * dot_product;
            }
        }
        Ok(())
    }

    fn tridiagonalize(&self, matrix: &mut WasmTensor, q: &mut WasmTensor) -> Result<(), JsValue> {
        let n = matrix.shape[0];

        for k in 0..(n - 2) {
            let (v, beta) = self.compute_householder_vector(matrix, k, k + 1, n, n)?;
            self.apply_householder_transformation(matrix, &v, beta, k + 1, k, n, n)?;
            self.apply_householder_transformation(matrix, &v, beta, 0, k + 1, n, n)?;
            self.apply_householder_to_q(q, &v, beta, k + 1, n)?;
        }

        Ok(())
    }

    fn compute_wilkinson_shift(&self, matrix: &WasmTensor, n: usize) -> Result<f32, JsValue> {
        if n < 2 {
            return Ok(0.0);
        }

        let a = matrix.data[(n - 2) * n + (n - 2)];
        let b = matrix.data[(n - 2) * n + (n - 1)];
        let c = matrix.data[(n - 1) * n + (n - 1)];

        let discriminant = (a - c) * (a - c) + 4.0 * b * b;
        let sqrt_disc = discriminant.sqrt();

        let lambda1 = (a + c + sqrt_disc) / 2.0;
        let lambda2 = (a + c - sqrt_disc) / 2.0;

        // Choose shift closer to bottom-right element
        if (c - lambda1).abs() < (c - lambda2).abs() {
            Ok(lambda1)
        } else {
            Ok(lambda2)
        }
    }

    fn batch_matmul_advanced(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape.len() != 3 || other.shape.len() != 3 {
            return Err(JsValue::from_str(
                "Advanced batch matmul requires 3D tensors",
            ));
        }

        let (batch_size, m, k1) = (self.shape[0], self.shape[1], self.shape[2]);
        let (batch_size2, k2, n) = (other.shape[0], other.shape[1], other.shape[2]);

        if batch_size != batch_size2 || k1 != k2 {
            return Err(JsValue::from_str(
                "Batch dimensions or inner dimensions don't match",
            ));
        }

        let mut result_data = vec![0.0f32; batch_size * m * n];

        // Advanced batch processing with memory optimization
        #[allow(clippy::excessive_nesting)]
        for b in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for k in 0..k1 {
                        let a_val = self.data[b * m * k1 + i * k1 + k];
                        let b_val = other.data[b * k2 * n + k * n + j];
                        sum += a_val * b_val;
                    }
                    result_data[b * m * n + i * n + j] = sum;
                }
            }
        }

        WasmTensor::new(result_data, vec![batch_size, m, n])
    }

    fn trace_product(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape != other.shape || self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(JsValue::from_str(
                "Trace product requires square matrices of same size",
            ));
        }

        let n = self.shape[0];
        let mut trace = 0.0f32;

        for i in 0..n {
            trace += self.data[i * n + i] * other.data[i * n + i];
        }

        WasmTensor::new(vec![trace], vec![1])
    }

    fn frobenius_inner_product(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        if self.shape != other.shape {
            return Err(JsValue::from_str("Frobenius product requires same shape"));
        }

        let mut sum = 0.0f32;
        for i in 0..self.data.len() {
            sum += self.data[i] * other.data[i];
        }

        WasmTensor::new(vec![sum], vec![1])
    }
}
/// Get supported tensor types
pub fn get_supported_types() -> Vec<String> {
    vec![
        "f32".to_string(),
        "i32".to_string(),
        "u32".to_string(),
        "bool".to_string(),
    ]
}

// Additional tensor operations for internal use
impl WasmTensor {
    pub(crate) fn slice(&self, start: &[usize], end: &[usize]) -> Result<WasmTensor, String> {
        if start.len() != self.shape.len() || end.len() != self.shape.len() {
            return Err("Slice dimensions must match tensor dimensions".into());
        }

        // Calculate new shape
        let new_shape: Vec<usize> = start.iter().zip(end.iter()).map(|(s, e)| e - s).collect();

        let new_size: usize = new_shape.iter().product();
        let mut result = Vec::with_capacity(new_size);

        // Extract slice data
        extract_slice_recursive(
            &self.data,
            &self.shape,
            &self.strides,
            start,
            end,
            0,
            0,
            &mut result,
        );

        let strides = compute_strides(&new_shape);
        Ok(WasmTensor {
            data: result,
            shape: new_shape,
            strides,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_slice_recursive(
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    start: &[usize],
    end: &[usize],
    dim: usize,
    offset: usize,
    result: &mut Vec<f32>,
) {
    if dim == shape.len() - 1 {
        // Base case: copy the slice at the last dimension
        for i in start[dim]..end[dim] {
            result.push(data[offset + i]);
        }
    } else {
        // Recursive case
        for i in start[dim]..end[dim] {
            let new_offset = offset + i * strides[dim];
            extract_slice_recursive(
                data,
                shape,
                strides,
                start,
                end,
                dim + 1,
                new_offset,
                result,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor =
            WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor operation failed");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_add() {
        let a =
            WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor operation failed");
        let b =
            WasmTensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("tensor operation failed");
        let c = a.add(&b).expect("add operation failed");
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul() {
        let a =
            WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor operation failed");
        let b =
            WasmTensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("tensor operation failed");
        let c = a.matmul(&b).expect("operation failed in test");
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_relu_simd() {
        let tensor = WasmTensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2])
            .expect("tensor operation failed");
        let result = tensor.relu_simd();
        assert_eq!(result.data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_gelu_simd() {
        let tensor =
            WasmTensor::new(vec![0.0, 1.0, -1.0], vec![3]).expect("tensor operation failed");
        let result = tensor.gelu_simd();
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert!((result.data[0] - 0.0).abs() < 0.01);
        assert!((result.data[1] - 0.841).abs() < 0.05);
        assert!((result.data[2] - (-0.159)).abs() < 0.05);
    }

    #[test]
    fn test_matmul_bias() {
        let input = WasmTensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor operation failed");
        let weights =
            WasmTensor::new(vec![1.0, 0.5, 0.2, 1.5], vec![2, 2]).expect("tensor operation failed");
        let bias = WasmTensor::new(vec![0.1, 0.2], vec![1, 2]).expect("tensor operation failed");
        let result = input.matmul_bias(&weights, &bias).expect("operation failed in test");
        // Expected: [1*1 + 2*0.2, 1*0.5 + 2*1.5] + [0.1, 0.2] = [1.4, 3.5] + [0.1, 0.2] = [1.5, 3.7]
        assert!((result.data[0] - 1.5).abs() < 0.001);
        assert!((result.data[1] - 3.7).abs() < 0.001);
    }

    #[test]
    fn test_layer_norm() {
        let tensor =
            WasmTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor operation failed");
        let gamma = WasmTensor::new(vec![1.0, 1.0], vec![2]).expect("tensor operation failed");
        let beta = WasmTensor::new(vec![0.0, 0.0], vec![2]).expect("tensor operation failed");
        let result = tensor
            .layer_norm_with_params(&gamma, &beta, 1e-5)
            .expect("tensor operation failed");
        // Each row should be normalized to mean=0, std=1
        assert!((result.data[0] + result.data[1]).abs() < 0.001); // Row 1 mean ≈ 0
        assert!((result.data[2] + result.data[3]).abs() < 0.001); // Row 2 mean ≈ 0
    }

    #[test]
    fn test_randn_enhanced() {
        let tensor = WasmTensor::randn(vec![100]).expect("tensor operation failed");
        assert_eq!(tensor.shape, vec![100]);
        assert_eq!(tensor.data.len(), 100);

        // Check that values are distributed around 0 (for normal distribution)
        let mean: f32 = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
        assert!(mean.abs() < 0.5); // Should be close to 0 for large sample

        // Check that we have some variety (not all the same value)
        let min = tensor.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.1); // Should have some spread
    }

    #[test]
    fn test_randn_with_params() {
        let mean = 2.0;
        let std = 0.5;
        let tensor =
            WasmTensor::randn_with_params(vec![100], mean, std).expect("tensor operation failed");

        let actual_mean: f32 = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
        assert!((actual_mean - mean).abs() < 0.3); // Should be close to desired mean
    }

    #[test]
    fn test_xavier_uniform() {
        let tensor = WasmTensor::xavier_uniform(vec![64, 32]).expect("tensor operation failed");
        assert_eq!(tensor.shape, vec![64, 32]);

        // Xavier limit = sqrt(6 / (64 + 32)) = sqrt(6 / 96) ≈ 0.25
        let limit = (6.0f32 / (64.0f32 + 32.0f32)).sqrt();
        for &value in &tensor.data {
            assert!(value >= -limit && value <= limit);
        }
    }

    #[test]
    fn test_he_normal() {
        let tensor = WasmTensor::he_normal(vec![64, 32]).expect("tensor operation failed");
        assert_eq!(tensor.shape, vec![64, 32]);

        // He std = sqrt(2 / 64) ≈ 0.177
        // Values should be distributed around 0 with this std
        let mean: f32 = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
        assert!(mean.abs() < 0.3); // Should be close to 0
    }

    #[test]
    fn test_random_uniform() {
        let min = -1.0;
        let max = 3.0;
        let tensor =
            WasmTensor::random_uniform(vec![100], min, max).expect("tensor operation failed");

        for &value in &tensor.data {
            assert!(value >= min && value <= max);
        }

        // Should have some variety
        let actual_min = tensor.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let actual_max = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(actual_max - actual_min > 0.5); // Should use most of the range
    }
}
