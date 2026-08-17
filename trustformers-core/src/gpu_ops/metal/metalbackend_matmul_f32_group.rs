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
    /// Row-major single-precision GEMM on the GPU: `C(m×n) = A(m×k) @ B(k×n)`.
    ///
    /// Runs through the Pure-Rust `oxicuda-metal` compute backend that this
    /// `MetalBackend` already owns (`self.mps_ops`, initialised once in
    /// [`MetalBackend::new`]). It used to construct **and `init()`** a brand new
    /// `oxicuda_metal::MetalBackend` on every call - a full Metal device, command
    /// queue and pipeline build per matmul - and then round-tripped the operands
    /// through `alloc`/`copy_htod`/`copy_dtoh`. Both costs are gone: the operands are
    /// uploaded straight into `StorageModeShared` buffers (GPU-resident on Apple
    /// Silicon's unified memory) and the GEMM runs against those in place.
    ///
    /// # Errors
    ///
    /// Returns a structured error when the shapes are degenerate, when the input
    /// slices do not match `m·k` / `k·n`, or when the oxicuda-metal backend failed to
    /// initialise - never a silently wrong result.
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        if m == 0 || k == 0 || n == 0 {
            return Err(TrustformersError::shape_error(format!(
                "matmul_f32 requires non-zero dimensions, got {m}x{k}x{n}"
            )));
        }
        if a.len() != m * k {
            return Err(TrustformersError::shape_error(format!(
                "matmul_f32: A has {} elements, expected m*k = {}",
                a.len(),
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(TrustformersError::shape_error(format!(
                "matmul_f32: B has {} elements, expected k*n = {}",
                b.len(),
                k * n
            )));
        }

        let oxi = self.mps_ops.as_ref().as_ref().ok_or_else(|| {
            TrustformersError::hardware_error(
                "oxicuda-metal compute backend is not initialised - GPU matmul unavailable",
                "MetalBackend::matmul_f32",
            )
        })?;

        // Upload operands into resident Shared buffers; allocate the result buffer.
        let a_buffer = Arc::new(self.create_buffer(a)?);
        let b_buffer = Arc::new(self.create_buffer(b)?);
        let c_buffer = Arc::new(self.device.new_buffer(
            (m * n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // oxicuda-metal dispatches on its own MTLCommandQueue; drain ours first so no
        // in-flight kernel of ours is still writing the operands.
        self.flush()?;
        super::metalbackend_initialize_mps_group::oxi_resident_gemm(
            oxi,
            &a_buffer,
            &b_buffer,
            &c_buffer,
            m,
            k,
            n,
            1.0_f64,
            "MetalBackend::matmul_f32",
        )?;

        let result_ptr = c_buffer.contents();
        if result_ptr.is_null() {
            return Err(TrustformersError::hardware_error(
                "GPU result buffer has no CPU mapping",
                "MetalBackend::matmul_f32",
            ));
        }
        // SAFETY: `c_buffer` is a Shared (CPU-mappable) allocation of exactly m*n f32,
        // and `oxi_resident_gemm` commits and waits before returning.
        let result =
            unsafe { std::slice::from_raw_parts(result_ptr as *const f32, m * n) }.to_vec();
        Ok(result)
    }

    /// Upload host data into a new `StorageModeShared` Metal buffer.
    pub(crate) fn create_buffer(&self, data: &[f32]) -> Result<Buffer> {
        let byte_size = std::mem::size_of_val(data) as u64;

        tracing::trace!(
            elements = data.len(),
            bytes = byte_size,
            "MetalBackend::create_buffer"
        );

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

        Ok(buffer)
    }
}
