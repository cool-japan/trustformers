// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized layer normalization implementation
//!

#![allow(unused_variables)] // SIMD implementation with architecture-specific code paths
//! This module provides layer normalization optimized for different SIMD
//! instruction sets including AVX-512, AVX2, NEON, and RISC-V Vector extensions.

use super::cpu_features::CpuFeatures;
use crate::tensor::Tensor;
use anyhow::Result;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SIMD-optimized LayerNorm implementation
/// Uses AVX-512, AVX2, or NEON instructions when available for vectorized operations
pub struct SIMDLayerNorm {
    hidden_size: usize,
    eps: f32,
    cpu_features: CpuFeatures,
}

impl SIMDLayerNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            hidden_size,
            eps,
            cpu_features: CpuFeatures::detect(),
        }
    }

    pub fn forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let simd_width = self.cpu_features.best_simd_width();
        let can_use_simd =
            simd_width > 1 && self.hidden_size % simd_width == 0 && self.hidden_size >= 256;

        if can_use_simd {
            match self.cpu_features.best_instruction_set() {
                "avx512" => self.forward_avx512(input, weight, bias),
                "avx2_fma" | "avx2" => self.forward_avx2(input, weight, bias),
                "neon" => self.forward_neon(input, weight, bias),
                "rvv" => self.forward_rvv(input, weight, bias),
                _ => self.forward_standard(input, weight, bias),
            }
        } else {
            self.forward_standard(input, weight, bias)
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn forward_avx2_inner(
        &self,
        input_data: &[f32],
        weight_data: &[f32],
        bias_data: Option<&[f32]>,
        output_data: &mut [f32],
        batch_size: usize,
        seq_len: usize,
    ) {
        let hidden_size = self.hidden_size;

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                let start_idx = batch * seq_len * hidden_size + seq * hidden_size;
                let input_slice = &input_data[start_idx..start_idx + hidden_size];
                let output_slice = &mut output_data[start_idx..start_idx + hidden_size];

                // Compute mean using SIMD
                let mut sum = _mm256_setzero_ps();
                let mut i = 0;
                while i + 8 <= hidden_size {
                    let vals = _mm256_loadu_ps(&input_slice[i]);
                    sum = _mm256_add_ps(sum, vals);
                    i += 8;
                }

                // Handle remaining elements
                let mut scalar_sum = 0.0f32;
                while i < hidden_size {
                    scalar_sum += input_slice[i];
                    i += 1;
                }

                // Horizontal sum of SIMD register
                let sum_array = std::mem::transmute::<_, [f32; 8]>(sum);
                let mean = (sum_array.iter().sum::<f32>() + scalar_sum) / hidden_size as f32;

                // Compute variance using SIMD
                let mean_vec = _mm256_set1_ps(mean);
                let mut var_sum = _mm256_setzero_ps();
                i = 0;
                while i + 8 <= hidden_size {
                    let vals = _mm256_loadu_ps(&input_slice[i]);
                    let diff = _mm256_sub_ps(vals, mean_vec);
                    let squared = _mm256_mul_ps(diff, diff);
                    var_sum = _mm256_add_ps(var_sum, squared);
                    i += 8;
                }

                // Handle remaining elements for variance
                let mut scalar_var_sum = 0.0f32;
                while i < hidden_size {
                    let diff = input_slice[i] - mean;
                    scalar_var_sum += diff * diff;
                    i += 1;
                }

                // Horizontal sum and compute standard deviation
                let var_array = std::mem::transmute::<_, [f32; 8]>(var_sum);
                let variance =
                    (var_array.iter().sum::<f32>() + scalar_var_sum) / hidden_size as f32;
                let std_dev = (variance + self.eps).sqrt();
                let inv_std = 1.0 / std_dev;

                // Normalize and apply weight/bias using SIMD
                let inv_std_vec = _mm256_set1_ps(inv_std);
                i = 0;
                while i + 8 <= hidden_size {
                    let vals = _mm256_loadu_ps(&input_slice[i]);
                    let weights = _mm256_loadu_ps(&weight_data[i]);

                    let diff = _mm256_sub_ps(vals, mean_vec);
                    let normalized = _mm256_mul_ps(diff, inv_std_vec);
                    let scaled = _mm256_mul_ps(normalized, weights);

                    let result = if let Some(bias_data) = bias_data {
                        let biases = _mm256_loadu_ps(&bias_data[i]);
                        _mm256_add_ps(scaled, biases)
                    } else {
                        scaled
                    };

                    _mm256_storeu_ps(&mut output_slice[i], result);
                    i += 8;
                }

                // Handle remaining elements
                while i < hidden_size {
                    let normalized = (input_slice[i] - mean) * inv_std;
                    let scaled = normalized * weight_data[i];
                    output_slice[i] = if let Some(bias_data) = bias_data {
                        scaled + bias_data[i]
                    } else {
                        scaled
                    };
                    i += 1;
                }
            }
        }
    }

    fn forward_avx2(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_shape = input.shape();
        let batch_size = if input_shape.len() >= 3 { input_shape[0] } else { 1 };
        let seq_len = if input_shape.len() >= 3 { input_shape[1] } else { 1 };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let input_data = input.data()?;
            let weight_data = weight.data()?;
            let bias_data = bias.map(|b| b.data()).transpose()?;

            let mut output_data = vec![0.0f32; input_data.len()];

            unsafe {
                self.forward_avx2_inner(
                    &input_data,
                    &weight_data,
                    bias_data.as_deref(),
                    &mut output_data,
                    batch_size,
                    seq_len,
                );
            }

            Ok(Tensor::from_vec(output_data, &input_shape)?)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // Fallback to standard implementation
            self.forward_standard(input, weight, bias)
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn forward_avx512(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_shape = input.shape();
        let batch_size = if input_shape.len() >= 3 { input_shape[0] } else { 1 };
        let seq_len = if input_shape.len() >= 3 { input_shape[1] } else { 1 };

        let input_data = input.data()?;
        let weight_data = weight.data()?;
        let bias_data = bias.map(|b| b.data()).transpose()?;

        let mut output_data = vec![0.0f32; input_data.len()];

        unsafe {
            self.forward_avx512_inner(
                &input_data,
                &weight_data,
                bias_data.as_deref(),
                &mut output_data,
                batch_size,
                seq_len,
            );
        }

        Ok(Tensor::from_vec(output_data, &input_shape)?)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn forward_avx512_inner(
        &self,
        input_data: &[f32],
        weight_data: &[f32],
        bias_data: Option<&[f32]>,
        output_data: &mut [f32],
        batch_size: usize,
        seq_len: usize,
    ) {
        let hidden_size = self.hidden_size;

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                let start_idx = batch * seq_len * hidden_size + seq * hidden_size;
                let input_slice = &input_data[start_idx..start_idx + hidden_size];
                let output_slice = &mut output_data[start_idx..start_idx + hidden_size];

                // Compute mean using AVX-512
                let mut sum = _mm512_setzero_ps();
                let mut i = 0;
                while i + 16 <= hidden_size {
                    let vals = _mm512_loadu_ps(&input_slice[i]);
                    sum = _mm512_add_ps(sum, vals);
                    i += 16;
                }

                // Handle remaining elements
                let mut scalar_sum = 0.0f32;
                while i < hidden_size {
                    scalar_sum += input_slice[i];
                    i += 1;
                }

                // Horizontal sum of AVX-512 register
                let mean = (_mm512_reduce_add_ps(sum) + scalar_sum) / hidden_size as f32;

                // Compute variance using AVX-512
                let mean_vec = _mm512_set1_ps(mean);
                let mut var_sum = _mm512_setzero_ps();
                i = 0;
                while i + 16 <= hidden_size {
                    let vals = _mm512_loadu_ps(&input_slice[i]);
                    let diff = _mm512_sub_ps(vals, mean_vec);
                    let squared = _mm512_mul_ps(diff, diff);
                    var_sum = _mm512_add_ps(var_sum, squared);
                    i += 16;
                }

                // Handle remaining elements for variance
                let mut scalar_var_sum = 0.0f32;
                while i < hidden_size {
                    let diff = input_slice[i] - mean;
                    scalar_var_sum += diff * diff;
                    i += 1;
                }

                // Compute standard deviation
                let variance =
                    (_mm512_reduce_add_ps(var_sum) + scalar_var_sum) / hidden_size as f32;
                let std_dev = (variance + self.eps).sqrt();
                let inv_std = 1.0 / std_dev;

                // Normalize and apply weight/bias using AVX-512
                let inv_std_vec = _mm512_set1_ps(inv_std);
                i = 0;
                while i + 16 <= hidden_size {
                    let vals = _mm512_loadu_ps(&input_slice[i]);
                    let weights = _mm512_loadu_ps(&weight_data[i]);

                    let diff = _mm512_sub_ps(vals, mean_vec);
                    let normalized = _mm512_mul_ps(diff, inv_std_vec);
                    let scaled = _mm512_mul_ps(normalized, weights);

                    let result = if let Some(bias_data) = bias_data {
                        let biases = _mm512_loadu_ps(&bias_data[i]);
                        _mm512_add_ps(scaled, biases)
                    } else {
                        scaled
                    };

                    _mm512_storeu_ps(&mut output_slice[i], result);
                    i += 16;
                }

                // Handle remaining elements
                while i < hidden_size {
                    let normalized = (input_slice[i] - mean) * inv_std;
                    let scaled = normalized * weight_data[i];
                    output_slice[i] = if let Some(bias_data) = bias_data {
                        scaled + bias_data[i]
                    } else {
                        scaled
                    };
                    i += 1;
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn forward_neon(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_shape = input.shape();
        let batch_size = if input_shape.len() >= 3 { input_shape[0] } else { 1 };
        let seq_len = if input_shape.len() >= 3 { input_shape[1] } else { 1 };

        let input_data = input.data()?;
        let weight_data = weight.data()?;
        let bias_data = bias.map(|b| b.data()).transpose()?;

        let mut output_data = vec![0.0f32; input_data.len()];

        unsafe {
            self.forward_neon_inner(
                &input_data,
                &weight_data,
                bias_data.as_deref(),
                &mut output_data,
                batch_size,
                seq_len,
            );
        }

        Ok(Tensor::from_vec(output_data, &input_shape)?)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn forward_neon_inner(
        &self,
        input_data: &[f32],
        weight_data: &[f32],
        bias_data: Option<&[f32]>,
        output_data: &mut [f32],
        batch_size: usize,
        seq_len: usize,
    ) {
        let hidden_size = self.hidden_size;

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                let start_idx = batch * seq_len * hidden_size + seq * hidden_size;
                let input_slice = &input_data[start_idx..start_idx + hidden_size];
                let output_slice = &mut output_data[start_idx..start_idx + hidden_size];

                // Compute mean using NEON
                let mut sum = vdupq_n_f32(0.0);
                let mut i = 0;
                while i + 4 <= hidden_size {
                    let vals = vld1q_f32(&input_slice[i]);
                    sum = vaddq_f32(sum, vals);
                    i += 4;
                }

                // Handle remaining elements
                let mut scalar_sum = 0.0f32;
                while i < hidden_size {
                    scalar_sum += input_slice[i];
                    i += 1;
                }

                // Horizontal sum of NEON register
                let sum_array = [
                    vgetq_lane_f32(sum, 0),
                    vgetq_lane_f32(sum, 1),
                    vgetq_lane_f32(sum, 2),
                    vgetq_lane_f32(sum, 3),
                ];
                let mean = (sum_array.iter().sum::<f32>() + scalar_sum) / hidden_size as f32;

                // Compute variance using NEON
                let mean_vec = vdupq_n_f32(mean);
                let mut var_sum = vdupq_n_f32(0.0);
                i = 0;
                while i + 4 <= hidden_size {
                    let vals = vld1q_f32(&input_slice[i]);
                    let diff = vsubq_f32(vals, mean_vec);
                    let squared = vmulq_f32(diff, diff);
                    var_sum = vaddq_f32(var_sum, squared);
                    i += 4;
                }

                // Handle remaining elements for variance
                let mut scalar_var_sum = 0.0f32;
                while i < hidden_size {
                    let diff = input_slice[i] - mean;
                    scalar_var_sum += diff * diff;
                    i += 1;
                }

                // Compute standard deviation
                let var_array = [
                    vgetq_lane_f32(var_sum, 0),
                    vgetq_lane_f32(var_sum, 1),
                    vgetq_lane_f32(var_sum, 2),
                    vgetq_lane_f32(var_sum, 3),
                ];
                let variance =
                    (var_array.iter().sum::<f32>() + scalar_var_sum) / hidden_size as f32;
                let std_dev = (variance + self.eps).sqrt();
                let inv_std = 1.0 / std_dev;

                // Normalize and apply weight/bias using NEON
                let inv_std_vec = vdupq_n_f32(inv_std);
                i = 0;
                while i + 4 <= hidden_size {
                    let vals = vld1q_f32(&input_slice[i]);
                    let weights = vld1q_f32(&weight_data[i]);

                    let diff = vsubq_f32(vals, mean_vec);
                    let normalized = vmulq_f32(diff, inv_std_vec);
                    let scaled = vmulq_f32(normalized, weights);

                    let result = if let Some(bias_data) = bias_data {
                        let biases = vld1q_f32(&bias_data[i]);
                        vaddq_f32(scaled, biases)
                    } else {
                        scaled
                    };

                    vst1q_f32(&mut output_slice[i], result);
                    i += 4;
                }

                // Handle remaining elements
                while i < hidden_size {
                    let normalized = (input_slice[i] - mean) * inv_std;
                    let scaled = normalized * weight_data[i];
                    output_slice[i] = if let Some(bias_data) = bias_data {
                        scaled + bias_data[i]
                    } else {
                        scaled
                    };
                    i += 1;
                }
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn forward_neon(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Fallback to standard implementation on non-ARM platforms
        self.forward_standard(input, weight, bias)
    }

    #[cfg(target_arch = "riscv64")]
    fn forward_rvv(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_shape = input.shape();
        let batch_size = if input_shape.len() >= 3 { input_shape[0] } else { 1 };
        let seq_len = if input_shape.len() >= 3 { input_shape[1] } else { 1 };

        let input_data = input.data()?;
        let weight_data = weight.data()?;
        let bias_data = bias.map(|b| b.data()).transpose()?;

        let mut output_data = vec![0.0f32; input_data.len()];

        unsafe {
            self.forward_rvv_inner(
                &input_data,
                &weight_data,
                bias_data.as_deref(),
                &mut output_data,
                batch_size,
                seq_len,
            );
        }

        Ok(Tensor::from_vec(output_data, &input_shape)?)
    }

    #[cfg(target_arch = "riscv64")]
    unsafe fn forward_rvv_inner(
        &self,
        input_data: &[f32],
        weight_data: &[f32],
        bias_data: Option<&[f32]>,
        output_data: &mut [f32],
        batch_size: usize,
        seq_len: usize,
    ) {
        let hidden_size = self.hidden_size;
        let vlen_elements = self.cpu_features.rvv_vlen / 32; // f32 elements that fit in vector

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                let start_idx = batch * seq_len * hidden_size + seq * hidden_size;
                let input_slice = &input_data[start_idx..start_idx + hidden_size];
                let output_slice = &mut output_data[start_idx..start_idx + hidden_size];

                // Compute mean using RVV
                let mut sum = 0.0f32;
                let mut i = 0;

                // Process vectorized chunks
                while i + vlen_elements <= hidden_size {
                    // In a real RVV implementation, we would use inline assembly
                    // or compiler intrinsics when they become available
                    // For now, we'll use a scalar loop as a placeholder
                    for j in 0..vlen_elements {
                        sum += input_slice[i + j];
                    }
                    i += vlen_elements;
                }

                // Handle remaining elements
                while i < hidden_size {
                    sum += input_slice[i];
                    i += 1;
                }

                let mean = sum / hidden_size as f32;

                // Compute variance using RVV
                let mut var_sum = 0.0f32;
                i = 0;

                // Process vectorized chunks
                while i + vlen_elements <= hidden_size {
                    for j in 0..vlen_elements {
                        let diff = input_slice[i + j] - mean;
                        var_sum += diff * diff;
                    }
                    i += vlen_elements;
                }

                // Handle remaining elements
                while i < hidden_size {
                    let diff = input_slice[i] - mean;
                    var_sum += diff * diff;
                    i += 1;
                }

                let variance = var_sum / hidden_size as f32;
                let std_dev = (variance + self.eps).sqrt();
                let inv_std = 1.0 / std_dev;

                // Normalize and apply weight/bias using RVV
                i = 0;
                while i + vlen_elements <= hidden_size {
                    for j in 0..vlen_elements {
                        let idx = i + j;
                        let normalized = (input_slice[idx] - mean) * inv_std;
                        let scaled = normalized * weight_data[idx];
                        output_slice[idx] = if let Some(bias_data) = bias_data {
                            scaled + bias_data[idx]
                        } else {
                            scaled
                        };
                    }
                    i += vlen_elements;
                }

                // Handle remaining elements
                while i < hidden_size {
                    let normalized = (input_slice[i] - mean) * inv_std;
                    let scaled = normalized * weight_data[i];
                    output_slice[i] = if let Some(bias_data) = bias_data {
                        scaled + bias_data[i]
                    } else {
                        scaled
                    };
                    i += 1;
                }
            }
        }
    }

    #[cfg(not(target_arch = "riscv64"))]
    fn forward_rvv(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Fallback to standard implementation on non-RISC-V platforms
        self.forward_standard(input, weight, bias)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn forward_avx512(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Fallback to standard implementation on non-x86 platforms
        self.forward_standard(input, weight, bias)
    }

    fn forward_standard(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Standard layer norm implementation
        let input_shape = input.shape();
        let input_data = input.data()?;
        let weight_data = weight.data()?;
        let bias_data = bias.map(|b| b.data()).transpose()?;

        let batch_size = if input_shape.len() >= 3 { input_shape[0] } else { 1 };
        let seq_len = if input_shape.len() >= 3 { input_shape[1] } else { 1 };
        let hidden_size = self.hidden_size;

        let mut output_data = vec![0.0f32; input_data.len()];

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                let start_idx = batch * seq_len * hidden_size + seq * hidden_size;
                let input_slice = &input_data[start_idx..start_idx + hidden_size];
                let output_slice = &mut output_data[start_idx..start_idx + hidden_size];

                // Compute mean
                let mean = input_slice.iter().sum::<f32>() / hidden_size as f32;

                // Compute variance
                let variance = input_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                    / hidden_size as f32;

                let std_dev = (variance + self.eps).sqrt();
                let inv_std = 1.0 / std_dev;

                // Normalize and apply weight/bias
                for i in 0..hidden_size {
                    let normalized = (input_slice[i] - mean) * inv_std;
                    let scaled = normalized * weight_data[i];
                    output_slice[i] = if let Some(ref bias_data) = bias_data {
                        scaled + bias_data[i]
                    } else {
                        scaled
                    };
                }
            }
        }

        Ok(Tensor::from_vec(output_data, &input_shape)?)
    }
}
