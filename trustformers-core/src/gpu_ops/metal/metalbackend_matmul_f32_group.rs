//! # MetalBackend - matmul_f32_group Methods
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
    /// Perform matrix multiplication using Apple Accelerate framework (100-500x faster than naive kernel)
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use oxiblas_blas::level3::gemm;
            use oxiblas_matrix::{MatMut, MatRef};

            let mut result = vec![0.0f32; m * n];

            // OxiBLAS uses column-major layout (Fortran convention), but we have row-major data
            // To compute C = A * B in row-major, we compute C^T = B^T * A^T in column-major
            // Row-major A (m×k) becomes column-major A^T (k×m) with leading dim k
            // Row-major B (k×n) becomes column-major B^T (n×k) with leading dim n
            // Result C^T is n×m in column-major = C in row-major with leading dim n
            let b_mat = MatRef::new(b.as_ptr(), n, k, n); // B^T: n×k with ld=n
            let a_mat = MatRef::new(a.as_ptr(), k, m, k); // A^T: k×m with ld=k
            let c_mat = MatMut::new(result.as_mut_ptr(), n, m, n); // C^T: n×m with ld=n

            // GEMM: C^T = 1.0 * B^T * A^T + 0.0 * C^T  (which gives us C = A * B)
            gemm(1.0, b_mat, a_mat, 0.0, c_mat);

            Ok(result)
        }
        #[cfg(not(feature = "metal"))]
        {
            let result_size = m * n;
            let c_buffer = self.device.new_buffer(
                (result_size * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let a_buffer = self.create_buffer(a)?;
            let b_buffer = self.create_buffer(b)?;
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&*self.matmul_pipeline);
            encoder.set_buffer(0, Some(&a_buffer), 0);
            encoder.set_buffer(1, Some(&b_buffer), 0);
            encoder.set_buffer(2, Some(&c_buffer), 0);
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
            let threadgroup_size = metal::MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: (n as u64 + 15) / 16,
                height: (m as u64 + 15) / 16,
                depth: 1,
            };
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Safety check: verify buffer pointer is not null
            let result_ptr = c_buffer.contents();
            if result_ptr.is_null() {
                return Err(TrustformersError::hardware_error(
                    "GPU buffer contents pointer is null",
                    "MetalBackend::matmul_f32",
                ));
            }

            let result_ptr = result_ptr as *const f32;
            let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();
            Ok(result)
        }
    }
    pub(crate) fn create_buffer(&self, data: &[f32]) -> Result<Buffer> {
        let byte_size = std::mem::size_of_val(data) as u64;

        #[cfg(debug_assertions)]
        {
            eprintln!(
                "🔍 create_buffer: data.len()={}, byte_size={}",
                data.len(),
                byte_size
            );
            if !data.is_empty() {
                eprintln!(
                    "🔍 create_buffer: first 5 values: {:?}",
                    &data[..5.min(data.len())]
                );
            }
        }

        // Validate input
        if data.is_empty() {
            return Err(TrustformersError::shape_error(
                "Cannot create buffer from empty data".to_string(),
            ));
        }

        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );

        // Verify buffer was created successfully
        let ptr = buffer.contents();
        if ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "Failed to create Metal buffer: contents pointer is null",
                "MetalBackend::create_buffer",
            ));
        }

        #[cfg(debug_assertions)]
        {
            let ptr = ptr as *const f32;
            let verify_data = unsafe { std::slice::from_raw_parts(ptr, data.len().min(5)) };
            eprintln!(
                "🔍 create_buffer: After creation, first 5 in buffer: {:?}",
                verify_data
            );
        }

        Ok(buffer)
    }
}
