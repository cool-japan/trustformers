//! Advanced fused GPU kernels for high-performance transformer operations
//!
//! This module implements sophisticated fused kernels that combine multiple operations
//! to minimize memory bandwidth requirements and maximize GPU utilization.

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use crate::errors::{Result, TrustformersError};

/// Fused LayerNorm + Linear operation
///
/// This kernel fuses three operations into one:
/// 1. Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
/// 2. Matrix multiplication: z = y * W^T
/// 3. Bias addition: out = z + bias
///
/// This fusion significantly reduces memory bandwidth by avoiding intermediate materialization.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub mod fused_layernorm_linear {
    use super::*;
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

    const PTX_KERNEL: &str = r#"
        .version 7.0
        .target sm_75
        .address_size 64

        // Fused LayerNorm + Linear kernel
        // This kernel performs:
        // 1. Compute mean and variance across features
        // 2. Normalize: (x - mean) / sqrt(var + eps)
        // 3. Scale and shift: * gamma + beta
        // 4. Matrix multiply with weight matrix
        // 5. Add bias
        //
        // Grid: (batch_size, output_features / BLOCK_SIZE)
        // Block: (BLOCK_SIZE, 1, 1)

        .visible .entry fused_layernorm_linear_kernel(
            .param .u64 input_ptr,          // Input tensor [batch, input_features]
            .param .u64 gamma_ptr,          // LayerNorm scale [input_features]
            .param .u64 beta_ptr,           // LayerNorm bias [input_features]
            .param .u64 weight_ptr,         // Linear weight [output_features, input_features]
            .param .u64 bias_ptr,           // Linear bias [output_features]
            .param .u64 output_ptr,         // Output [batch, output_features]
            .param .f32 epsilon,            // LayerNorm epsilon
            .param .u32 batch_size,
            .param .u32 input_features,
            .param .u32 output_features
        )
        {
            .reg .pred %p<4>;
            .reg .f32 %f<64>;
            .reg .b32 %r<32>;
            .reg .b64 %rd<32>;

            // Shared memory for reduction
            .shared .align 8 .b8 shared_mem[2048];  // 512 floats

            // Thread and block indices
            mov.u32 %r1, %tid.x;            // thread_x
            mov.u32 %r2, %ctaid.x;          // batch_idx
            mov.u32 %r3, %ctaid.y;          // output_block_idx

            // Load parameters
            ld.param.u32 %r4, [batch_size];
            ld.param.u32 %r5, [input_features];
            ld.param.u32 %r6, [output_features];
            ld.param.f32 %f1, [epsilon];

            // Check bounds
            setp.ge.u32 %p0, %r2, %r4;
            @%p0 bra EXIT;

            // Calculate input row offset: batch_idx * input_features
            mul.lo.u32 %r7, %r2, %r5;
            ld.param.u64 %rd1, [input_ptr];
            mul.wide.u32 %rd2, %r7, 4;      // * sizeof(float)
            add.u64 %rd3, %rd1, %rd2;       // input_row_ptr

            // ===== Phase 1: Compute mean =====
            mov.f32 %f2, 0.0;               // sum accumulator
            mov.u32 %r8, %r1;               // feature_idx = thread_x

            MEAN_LOOP:
                setp.ge.u32 %p1, %r8, %r5;
                @%p1 bra MEAN_LOOP_END;

                // Load input[batch_idx, feature_idx]
                mul.wide.u32 %rd4, %r8, 4;
                add.u64 %rd5, %rd3, %rd4;
                ld.global.f32 %f3, [%rd5];

                // Accumulate
                add.f32 %f2, %f2, %f3;

                add.u32 %r8, %r8, 256;      // blockDim.x stride
                bra MEAN_LOOP;
            MEAN_LOOP_END:

            // Store partial sum to shared memory
            mul.lo.u32 %r9, %r1, 4;
            mov.u64 %rd6, shared_mem;
            add.u64 %rd7, %rd6, %r9;
            st.shared.f32 [%rd7], %f2;
            bar.sync 0;

            // Thread 0 performs final reduction
            setp.ne.u32 %p2, %r1, 0;
            @%p2 bra SKIP_MEAN_REDUCE;

            mov.f32 %f4, 0.0;
            mov.u32 %r10, 0;
            MEAN_REDUCE:
                setp.ge.u32 %p3, %r10, 256;
                @%p3 bra MEAN_REDUCE_END;

                mul.lo.u32 %r11, %r10, 4;
                add.u64 %rd8, %rd6, %r11;
                ld.shared.f32 %f5, [%rd8];
                add.f32 %f4, %f4, %f5;

                add.u32 %r10, %r10, 1;
                bra MEAN_REDUCE;
            MEAN_REDUCE_END:

            // Compute mean = sum / input_features
            cvt.rn.f32.u32 %f6, %r5;
            div.rn.f32 %f7, %f4, %f6;      // mean
            st.shared.f32 [%rd6], %f7;      // Store mean to shared[0]

            SKIP_MEAN_REDUCE:
            bar.sync 0;
            ld.shared.f32 %f7, [%rd6];      // Load mean from shared[0]

            // ===== Phase 2: Compute variance =====
            mov.f32 %f8, 0.0;               // variance accumulator
            mov.u32 %r12, %r1;

            VAR_LOOP:
                setp.ge.u32 %p1, %r12, %r5;
                @%p1 bra VAR_LOOP_END;

                // Load input[batch_idx, feature_idx]
                mul.wide.u32 %rd9, %r12, 4;
                add.u64 %rd10, %rd3, %rd9;
                ld.global.f32 %f9, [%rd10];

                // Compute (x - mean)^2
                sub.f32 %f10, %f9, %f7;
                mul.f32 %f11, %f10, %f10;
                add.f32 %f8, %f8, %f11;

                add.u32 %r12, %r12, 256;
                bra VAR_LOOP;
            VAR_LOOP_END:

            // Store partial variance to shared memory
            add.u64 %rd11, %rd7, 0;
            st.shared.f32 [%rd11], %f8;
            bar.sync 0;

            // Thread 0 computes final variance
            @%p2 bra SKIP_VAR_REDUCE;

            mov.f32 %f12, 0.0;
            mov.u32 %r13, 0;
            VAR_REDUCE:
                setp.ge.u32 %p3, %r13, 256;
                @%p3 bra VAR_REDUCE_END;

                mul.lo.u32 %r14, %r13, 4;
                add.u64 %rd12, %rd6, %r14;
                ld.shared.f32 %f13, [%rd12];
                add.f32 %f12, %f12, %f13;

                add.u32 %r13, %r13, 1;
                bra VAR_REDUCE;
            VAR_REDUCE_END:

            // Compute stddev = sqrt(variance / input_features + epsilon)
            div.rn.f32 %f14, %f12, %f6;
            add.f32 %f15, %f14, %f1;
            sqrt.rn.f32 %f16, %f15;        // stddev

            // Compute reciprocal for normalization
            rcp.rn.f32 %f17, %f16;         // inv_stddev
            st.shared.f32 [%rd6 + 4], %f17; // Store to shared[1]

            SKIP_VAR_REDUCE:
            bar.sync 0;
            ld.shared.f32 %f17, [%rd6 + 4]; // Load inv_stddev

            // ===== Phase 3: LayerNorm + Linear =====
            // Compute output feature for this block
            mul.lo.u32 %r15, %r3, 256;
            add.u32 %r16, %r15, %r1;        // output_feature_idx

            setp.ge.u32 %p3, %r16, %r6;
            @%p3 bra EXIT;

            // Load weight and bias pointers
            ld.param.u64 %rd13, [weight_ptr];
            ld.param.u64 %rd14, [gamma_ptr];
            ld.param.u64 %rd15, [beta_ptr];
            ld.param.u64 %rd16, [bias_ptr];

            // Accumulator for dot product
            mov.f32 %f20, 0.0;
            mov.u32 %r17, 0;

            MATMUL_LOOP:
                setp.ge.u32 %p1, %r17, %r5;
                @%p1 bra MATMUL_LOOP_END;

                // Load and normalize input
                mul.wide.u32 %rd17, %r17, 4;
                add.u64 %rd18, %rd3, %rd17;
                ld.global.f32 %f21, [%rd18];   // input value

                // LayerNorm: (x - mean) * inv_stddev
                sub.f32 %f22, %f21, %f7;
                mul.f32 %f23, %f22, %f17;

                // Load gamma and beta
                add.u64 %rd19, %rd14, %rd17;
                ld.global.f32 %f24, [%rd19];   // gamma
                add.u64 %rd20, %rd15, %rd17;
                ld.global.f32 %f25, [%rd20];   // beta

                // Apply: normalized * gamma + beta
                fma.rn.f32 %f26, %f23, %f24, %f25;

                // Load weight[output_feature_idx, input_feature_idx]
                mul.lo.u32 %r18, %r16, %r5;
                add.u32 %r19, %r18, %r17;
                mul.wide.u32 %rd21, %r19, 4;
                add.u64 %rd22, %rd13, %rd21;
                ld.global.f32 %f27, [%rd22];   // weight

                // Accumulate: output += normalized_input * weight
                fma.rn.f32 %f20, %f26, %f27, %f20;

                add.u32 %r17, %r17, 1;
                bra MATMUL_LOOP;
            MATMUL_LOOP_END:

            // Add bias
            mul.wide.u32 %rd23, %r16, 4;
            add.u64 %rd24, %rd16, %rd23;
            ld.global.f32 %f28, [%rd24];
            add.f32 %f29, %f20, %f28;

            // Store result
            mul.lo.u32 %r20, %r2, %r6;      // batch_idx * output_features
            add.u32 %r21, %r20, %r16;       // + output_feature_idx
            mul.wide.u32 %rd25, %r21, 4;
            ld.param.u64 %rd26, [output_ptr];
            add.u64 %rd27, %rd26, %rd25;
            st.global.f32 [%rd27], %f29;

            EXIT:
            ret;
        }
    "#;

    /// Execute fused LayerNorm + Linear operation on CUDA
    pub fn execute(
        device: &CudaDevice,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch_size: usize,
        input_features: usize,
        output_features: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        // Load and compile kernel
        let module = device
            .load_ptx(
                PTX_KERNEL.into(),
                "fused_ln_linear",
                &["fused_layernorm_linear_kernel"],
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to load fused kernel: {}", e),
                    "fused_layernorm_linear",
                )
            })?;

        // Allocate device memory
        let input_dev = device.htod_copy(input.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input: {}", e),
                "fused_layernorm_linear",
            )
        })?;

        let gamma_dev = device.htod_copy(gamma.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy gamma: {}", e),
                "fused_layernorm_linear",
            )
        })?;

        let beta_dev = device.htod_copy(beta.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy beta: {}", e),
                "fused_layernorm_linear",
            )
        })?;

        let weight_dev = device.htod_copy(weight.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy weight: {}", e),
                "fused_layernorm_linear",
            )
        })?;

        let bias_dev = device.htod_copy(bias.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy bias: {}", e),
                "fused_layernorm_linear",
            )
        })?;

        let mut output_dev =
            device.alloc_zeros::<f32>(batch_size * output_features).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to allocate output: {}", e),
                    "fused_layernorm_linear",
                )
            })?;

        // Launch kernel
        let kernel = module.get_func("fused_layernorm_linear_kernel").ok_or_else(|| {
            TrustformersError::hardware_error("Kernel function not found", "fused_layernorm_linear")
        })?;

        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, (output_features as u32 + 255) / 256, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 2048,
        };

        unsafe {
            kernel
                .clone()
                .launch(
                    cfg,
                    (
                        &input_dev,
                        &gamma_dev,
                        &beta_dev,
                        &weight_dev,
                        &bias_dev,
                        &mut output_dev,
                        epsilon,
                        batch_size as u32,
                        input_features as u32,
                        output_features as u32,
                    ),
                )
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch kernel: {}", e),
                        "fused_layernorm_linear",
                    )
                })?;
        }

        // Copy result back
        let result = device.dtoh_sync_copy(&output_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result: {}", e),
                "fused_layernorm_linear",
            )
        })?;

        Ok(result)
    }
}

/// Fused Attention Softmax kernel with numerical stability
///
/// This kernel implements numerically stable softmax for attention scores:
/// 1. Find max value in each row (for numerical stability)
/// 2. Compute exp(x - max) for each element
/// 3. Sum exp values in each row
/// 4. Normalize: exp(x - max) / sum
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub mod fused_attention_softmax {
    use super::*;
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

    const PTX_KERNEL: &str = r#"
        .version 7.0
        .target sm_75
        .address_size 64

        .visible .entry fused_attention_softmax_kernel(
            .param .u64 input_ptr,
            .param .u64 output_ptr,
            .param .u64 mask_ptr,
            .param .u32 batch_size,
            .param .u32 num_heads,
            .param .u32 seq_length
        )
        {
            .reg .pred %p<4>;
            .reg .f32 %f<32>;
            .reg .b32 %r<24>;
            .reg .b64 %rd<16>;

            .shared .align 8 .b8 shared_mem[1024];

            // Get indices
            mov.u32 %r1, %tid.x;
            mov.u32 %r2, %ctaid.x;  // row index (batch * head * seq_len)

            ld.param.u32 %r3, [seq_length];
            ld.param.u64 %rd1, [input_ptr];
            ld.param.u64 %rd2, [output_ptr];
            ld.param.u64 %rd3, [mask_ptr];

            // Calculate row offset
            mul.lo.u32 %r4, %r2, %r3;
            mul.wide.u32 %rd4, %r4, 4;
            add.u64 %rd5, %rd1, %rd4;

            // Phase 1: Find max value (for numerical stability)
            mov.f32 %f1, 0xFF800000;  // -inf
            mov.u32 %r5, %r1;

            MAX_LOOP:
                setp.ge.u32 %p0, %r5, %r3;
                @%p0 bra MAX_LOOP_END;

                mul.wide.u32 %rd6, %r5, 4;
                add.u64 %rd7, %rd5, %rd6;
                ld.global.f32 %f2, [%rd7];

                // Check mask if provided
                setp.eq.u64 %p1, %rd3, 0;
                @%p1 bra SKIP_MASK1;

                add.u64 %rd8, %rd3, %rd6;
                ld.global.f32 %f3, [%rd8];
                setp.eq.f32 %p2, %f3, 0.0;
                @%p2 bra SKIP_MAX;

                SKIP_MASK1:
                max.f32 %f1, %f1, %f2;
                SKIP_MAX:

                add.u32 %r5, %r5, 256;
                bra MAX_LOOP;
            MAX_LOOP_END:

            // Reduce max across threads
            mov.u64 %rd9, shared_mem;
            mul.lo.u32 %r6, %r1, 4;
            add.u64 %rd10, %rd9, %r6;
            st.shared.f32 [%rd10], %f1;
            bar.sync 0;

            setp.ne.u32 %p3, %r1, 0;
            @%p3 bra SKIP_MAX_REDUCE;

            mov.f32 %f4, 0xFF800000;
            mov.u32 %r7, 0;
            MAX_REDUCE:
                setp.ge.u32 %p0, %r7, 256;
                @%p0 bra MAX_REDUCE_END;

                mul.lo.u32 %r8, %r7, 4;
                add.u64 %rd11, %rd9, %r8;
                ld.shared.f32 %f5, [%rd11];
                max.f32 %f4, %f4, %f5;

                add.u32 %r7, %r7, 1;
                bra MAX_REDUCE;
            MAX_REDUCE_END:
            st.shared.f32 [%rd9], %f4;

            SKIP_MAX_REDUCE:
            bar.sync 0;
            ld.shared.f32 %f6, [%rd9];  // max value

            // Phase 2: Compute exp(x - max) and sum
            mov.f32 %f7, 0.0;  // sum accumulator
            mov.u32 %r9, %r1;

            EXP_LOOP:
                setp.ge.u32 %p0, %r9, %r3;
                @%p0 bra EXP_LOOP_END;

                mul.wide.u32 %rd12, %r9, 4;
                add.u64 %rd13, %rd5, %rd12;
                ld.global.f32 %f8, [%rd13];

                // Compute exp(x - max)
                sub.f32 %f9, %f8, %f6;
                ex2.approx.ftz.f32 %f10, %f9;  // Fast exp approximation

                // Store exp value temporarily
                add.u64 %rd14, %rd2, %rd12;
                st.global.f32 [%rd14], %f10;

                // Accumulate sum
                add.f32 %f7, %f7, %f10;

                add.u32 %r9, %r9, 256;
                bra EXP_LOOP;
            EXP_LOOP_END:

            // Reduce sum across threads
            st.shared.f32 [%rd10], %f7;
            bar.sync 0;

            @%p3 bra SKIP_SUM_REDUCE;

            mov.f32 %f11, 0.0;
            mov.u32 %r10, 0;
            SUM_REDUCE:
                setp.ge.u32 %p0, %r10, 256;
                @%p0 bra SUM_REDUCE_END;

                mul.lo.u32 %r11, %r10, 4;
                add.u64 %rd15, %rd9, %r11;
                ld.shared.f32 %f12, [%rd15];
                add.f32 %f11, %f11, %f12;

                add.u32 %r10, %r10, 1;
                bra SUM_REDUCE;
            SUM_REDUCE_END:

            // Compute reciprocal
            rcp.rn.f32 %f13, %f11;
            st.shared.f32 [%rd9], %f13;

            SKIP_SUM_REDUCE:
            bar.sync 0;
            ld.shared.f32 %f14, [%rd9];  // 1 / sum

            // Phase 3: Normalize
            mov.u32 %r12, %r1;

            NORM_LOOP:
                setp.ge.u32 %p0, %r12, %r3;
                @%p0 bra NORM_LOOP_END;

                mul.wide.u32 %rd16, %r12, 4;
                add.u64 %rd17, %rd2, %rd16;
                ld.global.f32 %f15, [%rd17];

                // Normalize: exp / sum
                mul.f32 %f16, %f15, %f14;
                st.global.f32 [%rd17], %f16;

                add.u32 %r12, %r12, 256;
                bra NORM_LOOP;
            NORM_LOOP_END:

            ret;
        }
    "#;

    /// Execute fused attention softmax on CUDA
    pub fn execute(
        device: &CudaDevice,
        input: &[f32],
        mask: Option<&[f32]>,
        batch_size: usize,
        num_heads: usize,
        seq_length: usize,
    ) -> Result<Vec<f32>> {
        let total_rows = batch_size * num_heads * seq_length;

        // Load kernel
        let module = device
            .load_ptx(
                PTX_KERNEL.into(),
                "fused_softmax",
                &["fused_attention_softmax_kernel"],
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to load softmax kernel: {}", e),
                    "fused_attention_softmax",
                )
            })?;

        // Allocate device memory
        let input_dev = device.htod_copy(input.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input: {}", e),
                "fused_attention_softmax",
            )
        })?;

        let mut output_dev = device.alloc_zeros::<f32>(input.len()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output: {}", e),
                "fused_attention_softmax",
            )
        })?;

        let mask_dev = if let Some(mask_data) = mask {
            device.htod_copy(mask_data.to_vec()).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to copy mask: {}", e),
                    "fused_attention_softmax",
                )
            })?
        } else {
            device.alloc_zeros::<f32>(0).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to allocate dummy mask: {}", e),
                    "fused_attention_softmax",
                )
            })?
        };

        // Launch kernel
        let kernel = module.get_func("fused_attention_softmax_kernel").ok_or_else(|| {
            TrustformersError::hardware_error("Softmax kernel not found", "fused_attention_softmax")
        })?;

        let cfg = LaunchConfig {
            grid_dim: (total_rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 1024,
        };

        unsafe {
            kernel
                .clone()
                .launch(
                    cfg,
                    (
                        &input_dev,
                        &mut output_dev,
                        &mask_dev,
                        batch_size as u32,
                        num_heads as u32,
                        seq_length as u32,
                    ),
                )
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch softmax kernel: {}", e),
                        "fused_attention_softmax",
                    )
                })?;
        }

        // Copy result
        let result = device.dtoh_sync_copy(&output_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy softmax result: {}", e),
                "fused_attention_softmax",
            )
        })?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_fused_layernorm_linear() {
        use cudarc::driver::CudaDevice;

        // Skip if no CUDA device
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping test: no CUDA device");
                return;
            },
        };

        let batch_size = 2;
        let input_features = 4;
        let output_features = 3;

        // Create test inputs
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ];
        let gamma = vec![1.0; input_features];
        let beta = vec![0.0; input_features];
        let weight = vec![1.0; output_features * input_features];
        let bias = vec![0.0; output_features];

        let result = fused_layernorm_linear::execute(
            &device,
            &input,
            &gamma,
            &beta,
            &weight,
            &bias,
            batch_size,
            input_features,
            output_features,
            1e-5,
        );

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * output_features);
    }
}
