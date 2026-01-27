//! # MetalBackend - matmul_gelu_f32_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Fused matmul+GELU operation (GPU kernel fusion optimization)
    ///
    /// Combines matrix multiplication and GELU activation in a single kernel.
    /// This eliminates intermediate buffer allocation and one kernel dispatch overhead.
    ///
    /// # Performance Impact
    /// - Saves 1 kernel dispatch overhead (~10-20μs per call)
    /// - Eliminates 1 intermediate buffer allocation (M*N*sizeof(f32) bytes)
    /// - Reduces memory bandwidth by ~33% (1 write instead of write+read+write)
    ///
    /// For GPT-2 FFN (24 layers, 2 calls per layer = 48 total):
    /// - Saves 48 kernel dispatches per forward pass
    /// - Saves 48 intermediate buffers
    /// - Expected speedup: ~30% for FFN operations
    ///
    /// # Arguments
    /// * `a` - Input matrix [M, K]
    /// * `b` - Weight matrix [K, N]
    /// * `m` - Number of rows in A
    /// * `k` - Number of columns in A (= rows in B)
    /// * `n` - Number of columns in B
    ///
    /// # Returns
    /// Result vector of length M*N containing GELU(A @ B)
    pub fn matmul_gelu_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let result_size = m * n;

        // Create output buffer
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create input buffers
        let a_buffer = self.create_buffer(a)?;
        let b_buffer = self.create_buffer(b)?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&self.matmul_gelu_pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);

        // Set scalar parameters
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );

        // Dispatch thread groups (same as matmul - 16x16 tiles)
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64).div_ceil(16),
            height: (m as u64).div_ceil(16),
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed(); // Wait for GPU to complete

        // Safety check: verify buffer pointer is not null
        let result_ptr = c_buffer.contents();
        if result_ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "GPU buffer contents pointer is null",
                "MetalBackend::matmul_gelu_f32/matmul_bias_gelu_f32",
            ));
        }

        // Read result
        let result_ptr = result_ptr as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();

        Ok(result)
    }

    /// Fused matmul+bias+GELU operation (GPU kernel fusion optimization)
    ///
    /// Combines matrix multiplication, bias addition, and GELU activation in a single kernel.
    /// This eliminates TWO intermediate buffers and TWO kernel dispatches.
    ///
    /// # Performance Impact
    /// - Saves 2 kernel dispatch overheads (~20-40μs per call)
    /// - Eliminates 2 intermediate buffer allocations
    /// - Reduces memory bandwidth by ~50% (1 write instead of write+read+write+read+write)
    ///
    /// For GPT-2 FFN (24 layers, 2 calls per layer = 48 total):
    /// - Saves 96 kernel dispatches per forward pass
    /// - Saves 96 intermediate buffers
    /// - Expected speedup: ~40-50% for FFN operations
    ///
    /// # Arguments
    /// * `a` - Input matrix [M, K]
    /// * `b` - Weight matrix [K, N]
    /// * `bias` - Bias vector [N]
    /// * `m` - Number of rows in A
    /// * `k` - Number of columns in A (= rows in B)
    /// * `n` - Number of columns in B
    ///
    /// # Returns
    /// Result vector of length M*N containing GELU(A @ B + bias)
    pub fn matmul_bias_gelu_f32(
        &self,
        a: &[f32],
        b: &[f32],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let result_size = m * n;

        // Create output buffer
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create input buffers
        let a_buffer = self.create_buffer(a)?;
        let b_buffer = self.create_buffer(b)?;
        let bias_buffer = self.create_buffer(bias)?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&self.matmul_bias_gelu_pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&bias_buffer), 0);
        encoder.set_buffer(3, Some(&c_buffer), 0);

        // Set scalar parameters
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );

        // Dispatch thread groups (same as matmul - 16x16 tiles)
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64).div_ceil(16),
            height: (m as u64).div_ceil(16),
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed(); // Wait for GPU to complete

        // Safety check: verify buffer pointer is not null
        let result_ptr = c_buffer.contents();
        if result_ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "GPU buffer contents pointer is null",
                "MetalBackend::matmul_gelu_f32/matmul_bias_gelu_f32",
            ));
        }

        // Read result
        let result_ptr = result_ptr as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::errors::Result;
    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_fused_matmul_gelu_correctness() -> Result<()> {
        let backend = MetalBackend::new()?;

        // Simple 2x2 test case
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![0.5, 0.0, 0.0, 0.5]; // [2, 2] (identity-like)
        let m = 2;
        let k = 2;
        let n = 2;

        // Get fused result
        let fused_result = backend.matmul_gelu_f32(&a, &b, m, k, n)?;

        // Get separate results for comparison
        let matmul_result = backend.matmul_f32(&a, &b, m, k, n)?;
        let expected: Vec<f32> = matmul_result
            .iter()
            .map(|&x| {
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                if x > 10.0 {
                    x
                } else if x < -10.0 {
                    0.0
                } else {
                    let x_cubed = x * x * x;
                    let inner = 0.7978845608f32 * (x + 0.044715 * x_cubed);
                    let clamped = inner.clamp(-20.0, 20.0);
                    0.5 * x * (1.0 + clamped.tanh())
                }
            })
            .collect();

        // Compare results
        assert_eq!(fused_result.len(), expected.len());
        for (i, (&fused, &exp)) in fused_result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (fused - exp).abs() < 1e-5,
                "Mismatch at index {}: fused={} expected={}",
                i,
                fused,
                exp
            );
        }

        println!("✅ Fused matmul+GELU test passed!");
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_fused_matmul_gelu_performance() -> Result<()> {
        use std::time::Instant;

        let backend = MetalBackend::new()?;

        // Realistic size: GPT-2 FFN inner projection
        // [batch*seq, hidden] @ [hidden, 4*hidden]
        let m = 128; // batch * seq
        let k = 768; // hidden_size
        let n = 3072; // 4 * hidden_size

        // Random data
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        let a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Warmup
        let _ = backend.matmul_gelu_f32(&a, &b, m, k, n)?;

        // Benchmark fused version
        let start = Instant::now();
        for _ in 0..10 {
            let _ = backend.matmul_gelu_f32(&a, &b, m, k, n)?;
        }
        let fused_time = start.elapsed();

        // Benchmark separate version
        let start = Instant::now();
        for _ in 0..10 {
            let matmul_result = backend.matmul_f32(&a, &b, m, k, n)?;
            let _ = backend.gelu_f32(&matmul_result)?;
        }
        let separate_time = start.elapsed();

        println!(
            "⏱️  Fused:    {:?} (avg: {:?})",
            fused_time,
            fused_time / 10
        );
        println!(
            "⏱️  Separate: {:?} (avg: {:?})",
            separate_time,
            separate_time / 10
        );
        println!(
            "🚀 Speedup: {:.2}x",
            separate_time.as_secs_f64() / fused_time.as_secs_f64()
        );

        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_fused_matmul_bias_gelu_correctness() -> Result<()> {
        let backend = MetalBackend::new()?;

        // Simple 2x2 test case with bias
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![0.5, 0.0, 0.0, 0.5]; // [2, 2] (identity-like)
        let bias = vec![0.1, 0.2]; // [2]
        let m = 2;
        let k = 2;
        let n = 2;

        // Get fused result
        let fused_result = backend.matmul_bias_gelu_f32(&a, &b, &bias, m, k, n)?;

        // Get separate results for comparison
        let matmul_result = backend.matmul_f32(&a, &b, m, k, n)?;
        let expected: Vec<f32> = matmul_result
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                // Add bias
                let x_with_bias = x + bias[i % n];
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                if x_with_bias > 10.0 {
                    x_with_bias
                } else if x_with_bias < -10.0 {
                    0.0
                } else {
                    let x_cubed = x_with_bias * x_with_bias * x_with_bias;
                    let inner = 0.7978845608f32 * (x_with_bias + 0.044715 * x_cubed);
                    let clamped = inner.clamp(-20.0, 20.0);
                    0.5 * x_with_bias * (1.0 + clamped.tanh())
                }
            })
            .collect();

        // Compare results
        assert_eq!(fused_result.len(), expected.len());
        for (i, (&fused, &exp)) in fused_result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (fused - exp).abs() < 1e-5,
                "Mismatch at index {}: fused={} expected={}",
                i,
                fused,
                exp
            );
        }

        println!("✅ Fused matmul+bias+GELU test passed!");
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_fused_matmul_bias_gelu_performance() -> Result<()> {
        use std::time::Instant;

        let backend = MetalBackend::new()?;

        // Realistic size: GPT-2 FFN inner projection
        // [batch*seq, hidden] @ [hidden, 4*hidden] + bias[4*hidden]
        let m = 128; // batch * seq
        let k = 768; // hidden_size
        let n = 3072; // 4 * hidden_size

        // Random data
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        let a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();

        // Warmup
        let _ = backend.matmul_bias_gelu_f32(&a, &b, &bias, m, k, n)?;

        // Benchmark fused version (matmul+bias+GELU in one kernel)
        let start = Instant::now();
        for _ in 0..10 {
            let _ = backend.matmul_bias_gelu_f32(&a, &b, &bias, m, k, n)?;
        }
        let fused_time = start.elapsed();

        // Benchmark separate version (matmul, then add_bias, then GELU - 3 kernels)
        let start = Instant::now();
        for _ in 0..10 {
            let matmul_result = backend.matmul_f32(&a, &b, m, k, n)?;
            // Simulate bias addition (would be a separate kernel in real code)
            let bias_result: Vec<f32> =
                matmul_result.iter().enumerate().map(|(i, &x)| x + bias[i % n]).collect();
            let _ = backend.gelu_f32(&bias_result)?;
        }
        let separate_time = start.elapsed();

        println!(
            "⏱️  Fused (matmul+bias+GELU):    {:?} (avg: {:?})",
            fused_time,
            fused_time / 10
        );
        println!(
            "⏱️  Separate (3 operations):    {:?} (avg: {:?})",
            separate_time,
            separate_time / 10
        );
        println!(
            "🚀 Speedup: {:.2}x",
            separate_time.as_secs_f64() / fused_time.as_secs_f64()
        );

        Ok(())
    }
}
