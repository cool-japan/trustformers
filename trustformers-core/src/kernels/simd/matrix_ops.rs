//! SIMD-optimized matrix operations.
//!
//! This module provides SIMD-optimized matrix operations using scirs2-core's
//! `SimdUnifiedOps` trait. The trait automatically selects the best implementation
//! (AVX512, AVX2, NEON, etc.) based on platform capabilities.
//!
//! # Performance
//!
//! Uses scirs2-core's BLAS-accelerated GEMM which leverages:
//! - Accelerate framework on macOS
//! - Intel MKL on Intel systems
//! - OpenBLAS as fallback
//! - Platform-specific SIMD (AVX2/AVX-512/NEON) for smaller matrices

use super::cpu_features::CpuFeatures;
use crate::{Result, Tensor, TrustformersError};
use scirs2_core::ndarray::Array2;
use scirs2_core::simd_ops::SimdUnifiedOps;

// Keep legacy intrinsics for specialized low-level operations
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps,
    _mm512_setzero_ps, _mm512_storeu_ps,
};

pub struct SIMDMatrixOps {
    cpu_features: CpuFeatures,
}

impl Default for SIMDMatrixOps {
    fn default() -> Self {
        Self::new()
    }
}

impl SIMDMatrixOps {
    pub fn new() -> Self {
        Self {
            cpu_features: CpuFeatures::detect(),
        }
    }

    /// SIMD-optimized matrix multiplication using scirs2-core's BLAS-accelerated GEMM.
    ///
    /// This function uses scirs2-core's `SimdUnifiedOps::simd_gemm` which provides:
    /// - BLAS acceleration (Accelerate/MKL/OpenBLAS) for larger matrices
    /// - SIMD optimization (AVX2/AVX-512/NEON) for smaller matrices
    /// - Automatic platform detection and optimal backend selection
    ///
    /// Optimized for transformer feed-forward layers and attention projections.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Support 2D matrix multiplication (M, K) x (K, N) -> (M, N)
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                "Only 2D matrix multiplication supported",
                "matmul",
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if b_shape[0] != k {
            return Err(TrustformersError::tensor_op_error(
                "Matrix dimensions don't match for multiplication",
                "matmul",
            ));
        }

        // Use scirs2-core's BLAS-accelerated GEMM
        self.matmul_blas(a, b, m, k, n)
    }

    /// BLAS-accelerated matrix multiplication via scirs2-core's SimdUnifiedOps.
    ///
    /// This is the preferred implementation as it leverages:
    /// - Accelerate framework on macOS for optimal Apple Silicon/Intel performance
    /// - Intel MKL on Intel systems
    /// - OpenBLAS as fallback
    /// - For small matrices (<16 in any dimension), uses ndarray's optimized dot()
    fn matmul_blas(&self, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let a_data = a.data()?;
        let b_data = b.data()?;

        // Create Array2 views for scirs2-core's simd_gemm
        let a_arr = Array2::from_shape_vec((a.shape()[0], a.shape()[1]), a_data.to_vec())
            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
        let b_arr = Array2::from_shape_vec((b.shape()[0], b.shape()[1]), b_data.to_vec())
            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

        // For small matrices, use ndarray's optimized dot() method
        // simd_gemm has edge cases for very small matrices
        // Note: scirs2-core has bugs with matrices of size <64, use 64 as threshold
        const MIN_SIZE_FOR_SIMD_GEMM: usize = 64;
        let c_arr = if m < MIN_SIZE_FOR_SIMD_GEMM
            || n < MIN_SIZE_FOR_SIMD_GEMM
            || k < MIN_SIZE_FOR_SIMD_GEMM
        {
            a_arr.dot(&b_arr)
        } else {
            // Use BLAS-accelerated GEMM: C = alpha * A * B + beta * C
            let mut result = Array2::<f32>::zeros((m, n));
            f32::simd_gemm(1.0, &a_arr.view(), &b_arr.view(), 0.0, &mut result);
            result
        };

        // Convert back to Tensor
        let c_data: Vec<f32> = c_arr.into_raw_vec_and_offset().0;
        Tensor::from_vec(c_data, &[m, n])
    }

    /// Legacy SIMD-optimized matmul with manual platform dispatch.
    ///
    /// This is kept for cases where manual SIMD control is needed.
    /// For most use cases, prefer `matmul()` which uses BLAS acceleration.
    #[allow(dead_code)]
    pub fn matmul_legacy(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                "Only 2D matrix multiplication supported",
                "matmul",
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if b_shape[0] != k {
            return Err(TrustformersError::tensor_op_error(
                "Matrix dimensions don't match for multiplication",
                "matmul",
            ));
        }

        let simd_width = self.cpu_features.best_simd_width();
        let can_use_simd = simd_width > 1 && n % simd_width == 0 && n >= 64;

        if can_use_simd {
            match self.cpu_features.best_instruction_set() {
                "avx512" => self.matmul_avx512(a, b, m, k, n),
                "avx2_fma" | "avx2" => self.matmul_avx2(a, b, m, k, n),
                "neon" => self.matmul_neon(a, b, m, k, n),
                "rvv" => self.matmul_rvv(a, b, m, k, n),
                _ => self.matmul_standard(a, b, m, k, n),
            }
        } else {
            self.matmul_standard(a, b, m, k, n)
        }
    }

    fn matmul_standard(
        &self,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor> {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let mut c_data = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }

        Tensor::from_vec(c_data, &[m, n])
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn matmul_avx2(&self, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            self.matmul_avx2_inner(&a_data, &b_data, &mut c_data, m, k, n);
        }

        Ok(Tensor::from_vec(c_data, &[m, n])?)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn matmul_avx2_inner(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        c_data: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            for j in (0..n).step_by(8) {
                let mut sum = _mm256_setzero_ps();

                for l in 0..k {
                    let a_val = _mm256_set1_ps(a_data[i * k + l]);
                    let b_vec = _mm256_loadu_ps(&b_data[l * n + j]);
                    sum = _mm256_fmadd_ps(a_val, b_vec, sum);
                }

                _mm256_storeu_ps(&mut c_data[i * n + j], sum);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn matmul_avx512(
        &self,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor> {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            self.matmul_avx512_inner(&a_data, &b_data, &mut c_data, m, k, n);
        }

        Ok(Tensor::from_vec(c_data, &[m, n])?)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn matmul_avx512_inner(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        c_data: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            for j in (0..n).step_by(16) {
                let mut sum = _mm512_setzero_ps();

                for l in 0..k {
                    let a_val = _mm512_set1_ps(a_data[i * k + l]);
                    let b_vec = _mm512_loadu_ps(&b_data[l * n + j]);
                    sum = _mm512_fmadd_ps(a_val, b_vec, sum);
                }

                _mm512_storeu_ps(&mut c_data[i * n + j], sum);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn matmul_neon(&self, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            self.matmul_neon_inner(&a_data, &b_data, &mut c_data, m, k, n);
        }

        Tensor::from_vec(c_data, &[m, n])
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn matmul_neon_inner(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        c_data: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let mut sum = vdupq_n_f32(0.0);

                for l in 0..k {
                    let a_val = vdupq_n_f32(a_data[i * k + l]);
                    let b_vec = vld1q_f32(&b_data[l * n + j]);
                    sum = vfmaq_f32(sum, a_val, b_vec);
                }

                vst1q_f32(&mut c_data[i * n + j], sum);
            }
        }
    }

    // Fallback methods for non-supported platforms
    #[cfg(not(target_arch = "aarch64"))]
    fn matmul_neon(&self, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        self.matmul_standard(a, b, m, k, n)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn matmul_avx2(&self, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        self.matmul_standard(a, b, m, k, n)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn matmul_avx512(
        &self,
        a: &Tensor,
        b: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor> {
        self.matmul_standard(a, b, m, k, n)
    }

    #[cfg(target_arch = "riscv64")]
    fn matmul_rvv(&self, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            self.matmul_rvv_inner(&a_data, &b_data, &mut c_data, m, k, n);
        }

        Ok(Tensor::from_vec(c_data, &[m, n])?)
    }

    #[cfg(target_arch = "riscv64")]
    unsafe fn matmul_rvv_inner(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        c_data: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        let vlen_elements = self.cpu_features.rvv_vlen / 32; // f32 elements that fit in vector

        for i in 0..m {
            let mut j = 0;

            // Process vectorized chunks
            while j + vlen_elements <= n {
                // Initialize accumulator
                let mut sum = vec![0.0f32; vlen_elements];

                for l in 0..k {
                    let a_val = a_data[i * k + l];

                    // Vectorized multiply-accumulate
                    for v in 0..vlen_elements {
                        let b_val = b_data[l * n + j + v];
                        sum[v] += a_val * b_val;
                    }
                }

                // Store results
                for v in 0..vlen_elements {
                    c_data[i * n + j + v] = sum[v];
                }

                j += vlen_elements;
            }

            // Handle remaining elements
            while j < n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = sum;
                j += 1;
            }
        }
    }

    #[cfg(not(target_arch = "riscv64"))]
    fn matmul_rvv(&self, a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
        self.matmul_standard(a, b, m, k, n)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_exp_approx(x: __m256) -> __m256 {
    // Fast exp approximation using polynomial
    // exp(x) ≈ 1 + x + x²/2 + x³/6 (truncated Taylor series)
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let sixth = _mm256_set1_ps(1.0 / 6.0);

    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);

    let term1 = x;
    let term2 = _mm256_mul_ps(x2, half);
    let term3 = _mm256_mul_ps(x3, sixth);

    let result = _mm256_add_ps(one, term1);
    let result = _mm256_add_ps(result, term2);
    let result = _mm256_add_ps(result, term3);

    result
}
