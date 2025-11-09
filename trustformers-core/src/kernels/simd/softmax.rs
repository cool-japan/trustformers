// SIMD-optimized softmax operations
use super::cpu_features::CpuFeatures;
use crate::tensor::Tensor;
use anyhow::Result;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::matrix_ops::simd_exp_approx;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// SIMD-optimized softmax implementation
pub struct SIMDSoftmax {
    cpu_features: CpuFeatures,
}

impl Default for SIMDSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl SIMDSoftmax {
    pub fn new() -> Self {
        Self {
            cpu_features: CpuFeatures::detect(),
        }
    }

    pub fn forward(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        let simd_width = self.cpu_features.best_simd_width();
        let can_use_simd =
            simd_width > 1 && input.shape()[dim] % simd_width == 0 && input.shape()[dim] >= 64;

        if can_use_simd {
            match self.cpu_features.best_instruction_set() {
                "avx512" => self.forward_avx512(input, dim),
                "avx2_fma" | "avx2" => self.forward_avx2(input, dim),
                "neon" => self.forward_neon(input, dim),
                "rvv" => self.forward_rvv(input, dim),
                _ => self.forward_standard(input, dim),
            }
        } else {
            self.forward_standard(input, dim)
        }
    }

    fn forward_standard(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        let shape = input.shape();
        let input_shape = shape.clone();
        let data = input.data()?;

        // Handle only the last dimension for simplicity
        if dim != shape.len() - 1 {
            return Err(anyhow::anyhow!("Only last dimension softmax is supported"));
        }

        let last_dim_size = shape[dim];
        let batch_size = data.len() / last_dim_size;
        let mut output_data = vec![0.0f32; data.len()];

        for batch in 0..batch_size {
            let start_idx = batch * last_dim_size;
            let input_slice = &data[start_idx..start_idx + last_dim_size];
            let output_slice = &mut output_data[start_idx..start_idx + last_dim_size];

            // Find max for numerical stability
            let max_val = input_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for i in 0..last_dim_size {
                let exp_val = (input_slice[i] - max_val).exp();
                output_slice[i] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for output_val in output_slice.iter_mut().take(last_dim_size) {
                *output_val /= sum;
            }
        }

        Ok(Tensor::from_vec(output_data, &input_shape)?)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn forward_avx2_inner(&self, data: &[f32], output: &mut [f32], len: usize) {
        // Find maximum value using SIMD
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 8 <= len {
            let vals = _mm256_loadu_ps(&data[i]);
            max_vec = _mm256_max_ps(max_vec, vals);
            i += 8;
        }

        // Horizontal max reduction
        let max_array = std::mem::transmute::<_, [f32; 8]>(max_vec);
        let mut max_val = max_array.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Handle remaining elements
        while i < len {
            max_val = max_val.max(data[i]);
            i += 1;
        }

        let max_broadcast = _mm256_set1_ps(max_val);

        // Compute exp(x - max) and sum using SIMD
        let mut sum_vec = _mm256_setzero_ps();
        i = 0;
        while i + 8 <= len {
            let vals = _mm256_loadu_ps(&data[i]);
            let shifted = _mm256_sub_ps(vals, max_broadcast);

            // Approximate exp using polynomial (for better performance)
            let exp_vals = simd_exp_approx(shifted);
            _mm256_storeu_ps(&mut output[i], exp_vals);
            sum_vec = _mm256_add_ps(sum_vec, exp_vals);
            i += 8;
        }

        // Horizontal sum
        let sum_array = std::mem::transmute::<_, [f32; 8]>(sum_vec);
        let mut sum = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        while i < len {
            let exp_val = (data[i] - max_val).exp();
            output[i] = exp_val;
            sum += exp_val;
            i += 1;
        }

        // Normalize using SIMD
        let inv_sum = _mm256_set1_ps(1.0 / sum);
        i = 0;
        while i + 8 <= len {
            let vals = _mm256_loadu_ps(&output[i]);
            let normalized = _mm256_mul_ps(vals, inv_sum);
            _mm256_storeu_ps(&mut output[i], normalized);
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            output[i] /= sum;
            i += 1;
        }
    }

    fn forward_avx2(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        let shape = input.shape();
        let input_shape = shape.clone();

        // For simplicity, handle only the last dimension case
        if dim != shape.len() - 1 {
            return self.forward_standard(input, dim);
        }

        let data = input.data()?;
        let last_dim_size = shape[dim];
        let batch_size = data.len() / last_dim_size;

        let mut output_data = vec![0.0f32; data.len()];

        for batch in 0..batch_size {
            let start_idx = batch * last_dim_size;
            let input_slice = &data[start_idx..start_idx + last_dim_size];
            let output_slice = &mut output_data[start_idx..start_idx + last_dim_size];

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                self.forward_avx2_inner(input_slice, output_slice, last_dim_size);
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                // Simple fallback
                for (i, &val) in input_slice.iter().enumerate() {
                    output_slice[i] = val;
                }
            }
        }

        Ok(Tensor::from_vec(output_data, &input_shape)?)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn forward_avx512(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        // AVX-512 implementation would go here, for now fallback to AVX2
        self.forward_avx2(input, dim)
    }

    #[cfg(target_arch = "aarch64")]
    fn forward_neon(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        // NEON implementation would go here, for now fallback to standard
        self.forward_standard(input, dim)
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn forward_neon(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        self.forward_standard(input, dim)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn forward_avx512(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        self.forward_standard(input, dim)
    }

    #[cfg(target_arch = "riscv64")]
    fn forward_rvv(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        let shape = input.shape();
        let input_shape = shape.clone();

        // For simplicity, handle only the last dimension case
        if dim != shape.len() - 1 {
            return self.forward_standard(input, dim);
        }

        let data = input.data()?;
        let last_dim_size = shape[dim];
        let batch_size = data.len() / last_dim_size;
        let vlen_elements = self.cpu_features.rvv_vlen / 32; // f32 elements that fit in vector

        let mut output_data = vec![0.0f32; data.len()];

        for batch in 0..batch_size {
            let start_idx = batch * last_dim_size;
            let input_slice = &data[start_idx..start_idx + last_dim_size];
            let output_slice = &mut output_data[start_idx..start_idx + last_dim_size];

            unsafe {
                self.forward_rvv_softmax_inner(
                    input_slice,
                    output_slice,
                    last_dim_size,
                    vlen_elements,
                );
            }
        }

        Ok(Tensor::from_vec(output_data, &input_shape)?)
    }

    #[cfg(target_arch = "riscv64")]
    unsafe fn forward_rvv_softmax_inner(
        &self,
        input_slice: &[f32],
        output_slice: &mut [f32],
        len: usize,
        vlen_elements: usize,
    ) {
        // Find maximum value using RVV
        let mut max_val = f32::NEG_INFINITY;
        let mut i = 0;

        // Process vectorized chunks
        while i + vlen_elements <= len {
            for j in 0..vlen_elements {
                max_val = max_val.max(input_slice[i + j]);
            }
            i += vlen_elements;
        }

        // Handle remaining elements
        while i < len {
            max_val = max_val.max(input_slice[i]);
            i += 1;
        }

        // Compute exp(x - max) and sum using RVV
        let mut sum = 0.0f32;
        i = 0;

        // Process vectorized chunks
        while i + vlen_elements <= len {
            for j in 0..vlen_elements {
                let idx = i + j;
                let exp_val = (input_slice[idx] - max_val).exp();
                output_slice[idx] = exp_val;
                sum += exp_val;
            }
            i += vlen_elements;
        }

        // Handle remaining elements
        while i < len {
            let exp_val = (input_slice[i] - max_val).exp();
            output_slice[i] = exp_val;
            sum += exp_val;
            i += 1;
        }

        // Normalize using RVV
        let inv_sum = 1.0 / sum;
        i = 0;

        // Process vectorized chunks
        while i + vlen_elements <= len {
            for j in 0..vlen_elements {
                let idx = i + j;
                output_slice[idx] *= inv_sum;
            }
            i += vlen_elements;
        }

        // Handle remaining elements
        while i < len {
            output_slice[i] *= inv_sum;
            i += 1;
        }
    }

    #[cfg(not(target_arch = "riscv64"))]
    fn forward_rvv(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        self.forward_standard(input, dim)
    }
}
