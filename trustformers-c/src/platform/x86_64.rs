//! x86_64-specific optimizations for TrustformeRS-C
//!
//! This module provides SIMD optimizations using AVX2, FMA, and other x86_64 extensions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub mod avx2_optimizations {
    use super::*;

    /// Vectorized matrix multiplication using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn matrix_multiply_avx2_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        // AVX2 optimized matrix multiplication
        for i in (0..m).step_by(8) {
            for j in (0..n).step_by(8) {
                let mut sum = [_mm256_setzero_ps(); 8];

                for ki in (0..k).step_by(8) {
                    let a_vec = [
                        _mm256_loadu_ps(a.as_ptr().add(i * k + ki)),
                        _mm256_loadu_ps(a.as_ptr().add((i + 1) * k + ki)),
                        _mm256_loadu_ps(a.as_ptr().add((i + 2) * k + ki)),
                        _mm256_loadu_ps(a.as_ptr().add((i + 3) * k + ki)),
                        _mm256_loadu_ps(a.as_ptr().add((i + 4) * k + ki)),
                        _mm256_loadu_ps(a.as_ptr().add((i + 5) * k + ki)),
                        _mm256_loadu_ps(a.as_ptr().add((i + 6) * k + ki)),
                        _mm256_loadu_ps(a.as_ptr().add((i + 7) * k + ki)),
                    ];

                    let b_vec = [
                        _mm256_loadu_ps(b.as_ptr().add(ki * n + j)),
                        _mm256_loadu_ps(b.as_ptr().add((ki + 1) * n + j)),
                        _mm256_loadu_ps(b.as_ptr().add((ki + 2) * n + j)),
                        _mm256_loadu_ps(b.as_ptr().add((ki + 3) * n + j)),
                        _mm256_loadu_ps(b.as_ptr().add((ki + 4) * n + j)),
                        _mm256_loadu_ps(b.as_ptr().add((ki + 5) * n + j)),
                        _mm256_loadu_ps(b.as_ptr().add((ki + 6) * n + j)),
                        _mm256_loadu_ps(b.as_ptr().add((ki + 7) * n + j)),
                    ];

                    for row in 0..8 {
                        for col in 0..8 {
                            sum[row] = _mm256_fmadd_ps(a_vec[row], b_vec[col], sum[row]);
                        }
                    }
                }

                for row in 0..8 {
                    if i + row < m {
                        _mm256_storeu_ps(c.as_mut_ptr().add((i + row) * n + j), sum[row]);
                    }
                }
            }
        }
    }

    /// Vectorized dot product using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let len = a.len().min(b.len());

        for i in (0..len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }

        // Horizontal sum of 8 elements
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(sum_low, sum_high);

        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));

        _mm_cvtss_f32(sum32)
    }

    /// Vectorized softmax using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn softmax_avx2_f32(input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());

        // Find max value
        let mut max_val = f32::NEG_INFINITY;
        for &val in input {
            max_val = max_val.max(val);
        }
        let max_vec = _mm256_set1_ps(max_val);

        // Compute exp(x - max) and sum
        let mut sum = _mm256_setzero_ps();
        for i in (0..len).step_by(8) {
            let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
            let sub_vec = _mm256_sub_ps(input_vec, max_vec);

            // Fast exp approximation
            let exp_vec = exp_approx_avx2(sub_vec);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), exp_vec);
            sum = _mm256_add_ps(sum, exp_vec);
        }

        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(sum_low, sum_high);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        let total_sum = _mm_cvtss_f32(sum32);

        let inv_sum = _mm256_set1_ps(1.0 / total_sum);

        // Normalize
        for i in (0..len).step_by(8) {
            let exp_vec = _mm256_loadu_ps(output.as_ptr().add(i));
            let result = _mm256_mul_ps(exp_vec, inv_sum);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    /// Fast exponential approximation using AVX2
    #[target_feature(enable = "avx2")]
    unsafe fn exp_approx_avx2(x: __m256) -> __m256 {
        // Taylor series: exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
        let one = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(1.0 / 6.0);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c5 = _mm256_set1_ps(1.0 / 120.0);

        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let x4 = _mm256_mul_ps(x3, x);
        let x5 = _mm256_mul_ps(x4, x);

        let mut result = one;
        result = _mm256_fmadd_ps(x, one, result);
        result = _mm256_fmadd_ps(x2, c2, result);
        result = _mm256_fmadd_ps(x3, c3, result);
        result = _mm256_fmadd_ps(x4, c4, result);
        result = _mm256_fmadd_ps(x5, c5, result);

        result
    }
}

/// AVX2 matrix multiplication (safe wrapper)
pub fn avx2_matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            avx2_optimizations::matrix_multiply_avx2_f32(a, b, c, m, n, k);
        }
    } else {
        // Fallback to generic implementation
        generic_matrix_multiply(a, b, c, m, n, k);
    }
}

/// Generic matrix multiplication fallback
fn generic_matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for ki in 0..k {
                sum += a[i * k + ki] * b[ki * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// x86_64 specific optimizer
pub struct X86_64Optimizer {
    has_avx2: bool,
    has_fma: bool,
    has_avx512: bool,
}

impl X86_64Optimizer {
    pub fn new() -> Self {
        Self {
            has_avx2: is_x86_feature_detected!("avx2"),
            has_fma: is_x86_feature_detected!("fma"),
            has_avx512: is_x86_feature_detected!("avx512f"),
        }
    }

    pub fn get_features(&self) -> CpuFeatures {
        CpuFeatures {
            has_avx2: self.has_avx2,
            has_fma: self.has_fma,
            has_avx512: self.has_avx512,
            has_sse41: is_x86_feature_detected!("sse4.1"),
        }
    }

    pub fn matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        if self.has_avx2 {
            unsafe {
                avx2_optimizations::matrix_multiply_avx2_f32(a, b, c, m, n, k);
            }
        } else {
            generic_matrix_multiply(a, b, c, m, n, k);
        }
    }

    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.has_avx2 {
            unsafe { avx2_optimizations::dot_product_avx2_f32(a, b) }
        } else {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
    }

    pub fn softmax(&self, input: &[f32], output: &mut [f32]) {
        if self.has_avx2 {
            unsafe {
                avx2_optimizations::softmax_avx2_f32(input, output);
            }
        } else {
            // Generic softmax implementation
            let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;

            for (i, &x) in input.iter().enumerate() {
                let exp_val = (x - max_val).exp();
                output[i] = exp_val;
                sum += exp_val;
            }

            for val in output.iter_mut() {
                *val /= sum;
            }
        }
    }
}

/// CPU features for x86_64
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_fma: bool,
    pub has_avx512: bool,
    pub has_sse41: bool,
}

/// C API exports for x86_64 optimizations
#[no_mangle]
pub extern "C" fn trustformers_x86_64_get_features() -> CpuFeatures {
    X86_64Optimizer::new().get_features()
}

#[no_mangle]
pub extern "C" fn trustformers_x86_64_matrix_multiply(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    if a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    unsafe {
        let a_slice = std::slice::from_raw_parts(a, m * k);
        let b_slice = std::slice::from_raw_parts(b, k * n);
        let c_slice = std::slice::from_raw_parts_mut(c, m * n);

        let optimizer = X86_64Optimizer::new();
        optimizer.matrix_multiply(a_slice, b_slice, c_slice, m, n, k);
    }
}

#[no_mangle]
pub extern "C" fn trustformers_x86_64_dot_product(a: *const f32, b: *const f32, len: usize) -> f32 {
    if a.is_null() || b.is_null() {
        return 0.0;
    }

    unsafe {
        let a_slice = std::slice::from_raw_parts(a, len);
        let b_slice = std::slice::from_raw_parts(b, len);

        let optimizer = X86_64Optimizer::new();
        optimizer.dot_product(a_slice, b_slice)
    }
}
