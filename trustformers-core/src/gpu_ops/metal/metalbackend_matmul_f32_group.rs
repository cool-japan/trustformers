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
            // Route the GEMM through the Pure-Rust oxicuda Metal backend via the
            // high-level `ComputeBackend` trait. Import the oxicuda backend under
            // an alias to avoid clashing with `super::common::*`'s
            // `metal::Device as MetalDevice`.
            use oxicuda_backend::{BackendTranspose, ComputeBackend};
            use oxicuda_metal::MetalBackend as OxiMetalBackend;

            // Row-major f32 host data → little-endian byte buffers for upload.
            let a_bytes = f32_slice_to_le_bytes(a);
            let b_bytes = f32_slice_to_le_bytes(b);
            let c_len_bytes = m * n * 4;

            let mut backend = OxiMetalBackend::new();
            backend.init().map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal init: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;

            let a_h = backend.alloc(a_bytes.len()).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal alloc a: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;
            let b_h = backend.alloc(b_bytes.len()).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal alloc b: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;
            let c_h = backend.alloc(c_len_bytes).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal alloc c: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;

            backend.copy_htod(a_h, &a_bytes).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal htod a: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;
            backend.copy_htod(b_h, &b_bytes).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal htod b: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;
            backend.copy_htod(c_h, &vec![0u8; c_len_bytes]).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal htod c: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;

            // C(m×n) = 1.0 * A(m×k) * B(k×n) + 0.0 * C, all row-major.
            // lda=k, ldb=n, ldc=n.
            backend
                .gemm(
                    BackendTranspose::NoTrans,
                    BackendTranspose::NoTrans,
                    m,
                    n,
                    k,
                    1.0_f64,
                    a_h,
                    k,
                    b_h,
                    n,
                    0.0_f64,
                    c_h,
                    n,
                )
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("oxicuda-metal gemm: {e}"),
                        "MetalBackend::matmul_f32",
                    )
                })?;

            let mut out_bytes = vec![0u8; c_len_bytes];
            backend.copy_dtoh(&mut out_bytes, c_h).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal dtoh: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;
            let result = le_bytes_to_f32_vec(&out_bytes);

            backend.free(a_h).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal free a: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;
            backend.free(b_h).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal free b: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;
            backend.free(c_h).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("oxicuda-metal free c: {e}"),
                    "MetalBackend::matmul_f32",
                )
            })?;

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

#[cfg(all(target_os = "macos", feature = "metal"))]
fn f32_slice_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn le_bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(buf));
    }
    out
}
