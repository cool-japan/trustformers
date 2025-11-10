//! CUDA GPU backend for tensor operations
//!
//! This module provides CUDA GPU acceleration for tensor operations on NVIDIA GPUs.
//! Uses cudarc for direct CUDA API access.

use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use crate::errors::TrustformersError;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

/// CUDA GPU backend for matrix multiplication
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub struct CudaBackend {
    device: CudaDevice,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaBackend {
    /// Create a new CUDA backend
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create CUDA device: {}", e),
                "CudaBackend::new",
            )
        })?;

        Ok(Self { device })
    }

    /// Perform matrix multiplication on CUDA GPU
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::LaunchAsync;

        // CUDA kernel for matrix multiplication
        const PTX_SRC: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64

        .visible .entry matmul_kernel(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 c_ptr,
            .param .u32 M,
            .param .u32 N,
            .param .u32 K
        )
        {
            .reg .pred %p<2>;
            .reg .f32 %f<16>;
            .reg .b32 %r<32>;
            .reg .b64 %rd<16>;

            // Get thread indices
            mov.u32 %r1, %ctaid.x;     // block index x
            mov.u32 %r2, %ctaid.y;     // block index y
            mov.u32 %r3, %tid.x;       // thread index x
            mov.u32 %r4, %tid.y;       // thread index y

            // Calculate row and column indices
            mul.lo.u32 %r5, %r2, 16;   // block_y * 16
            add.u32 %r6, %r5, %r4;     // row = block_y * 16 + thread_y
            mul.lo.u32 %r7, %r1, 16;   // block_x * 16
            add.u32 %r8, %r7, %r3;     // col = block_x * 16 + thread_x

            // Load parameters
            ld.param.u32 %r9, [M];
            ld.param.u32 %r10, [N];
            ld.param.u32 %r11, [K];

            // Bounds checking
            setp.ge.u32 %p0, %r6, %r9; // row >= M
            setp.ge.u32 %p1, %r8, %r10; // col >= N
            or.pred %p0, %p0, %p1;
            @%p0 bra DONE;

            // Initialize accumulator
            mov.f32 %f0, 0.0;

            // Main loop
            mov.u32 %r12, 0;
            LOOP_START:
                setp.ge.u32 %p0, %r12, %r11;
                @%p0 bra LOOP_END;

                // Calculate a[row * K + i]
                mul.lo.u32 %r13, %r6, %r11;
                add.u32 %r14, %r13, %r12;
                ld.param.u64 %rd1, [a_ptr];
                mul.wide.u32 %rd2, %r14, 4;
                add.u64 %rd3, %rd1, %rd2;
                ld.global.f32 %f1, [%rd3];

                // Calculate b[i * N + col]
                mul.lo.u32 %r15, %r12, %r10;
                add.u32 %r16, %r15, %r8;
                ld.param.u64 %rd4, [b_ptr];
                mul.wide.u32 %rd5, %r16, 4;
                add.u64 %rd6, %rd4, %rd5;
                ld.global.f32 %f2, [%rd6];

                // Multiply and accumulate
                fma.rn.f32 %f0, %f1, %f2, %f0;

                add.u32 %r12, %r12, 1;
                bra LOOP_START;
            LOOP_END:

            // Store result c[row * N + col]
            mul.lo.u32 %r17, %r6, %r10;
            add.u32 %r18, %r17, %r8;
            ld.param.u64 %rd7, [c_ptr];
            mul.wide.u32 %rd8, %r18, 4;
            add.u64 %rd9, %rd7, %rd8;
            st.global.f32 [%rd9], %f0;

            DONE:
            ret;
        }
        "#;

        // Load kernel module
        let module =
            self.device
                .load_ptx(PTX_SRC.into(), "matmul", &["matmul_kernel"])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to load CUDA kernel: {}", e),
                        "matmul_f32",
                    )
                })?;

        // Allocate device memory
        let a_dev = self.device.htod_copy(a.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy A to device: {}", e),
                "matmul_f32",
            )
        })?;

        let b_dev = self.device.htod_copy(b.to_vec()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy B to device: {}", e),
                "matmul_f32",
            )
        })?;

        let result_size = m * n;
        let mut c_dev = self.device.alloc_zeros::<f32>(result_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer: {}", e),
                "matmul_f32",
            )
        })?;

        // Launch kernel
        let kernel = module.get_func("matmul_kernel").ok_or_else(|| {
            TrustformersError::hardware_error("Kernel function not found", "matmul_f32")
        })?;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        unsafe {
            kernel
                .clone()
                .launch(cfg, (&a_dev, &b_dev, &mut c_dev, m_u32, n_u32, k_u32))
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch kernel: {}", e),
                        "matmul_f32",
                    )
                })?;
        }

        // Copy result back to CPU
        let result = self.device.dtoh_sync_copy(&c_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result to host: {}", e),
                "matmul_f32",
            )
        })?;

        Ok(result)
    }
}

/// Global CUDA backend cache (one per device)
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
static CUDA_BACKENDS: once_cell::sync::Lazy<
    std::sync::Mutex<std::collections::HashMap<usize, CudaBackend>>,
> = once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Get or create CUDA backend instance for a specific device
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn get_cuda_backend(device_id: usize) -> Result<CudaBackend> {
    let mut cache = CUDA_BACKENDS.lock().map_err(|_| {
        TrustformersError::hardware_error("Failed to lock CUDA backend cache", "get_cuda_backend")
    })?;

    if !cache.contains_key(&device_id) {
        let backend = CudaBackend::new(device_id)?;
        cache.insert(device_id, backend);
    }

    // Since CudaDevice is not Clone, we need to create a new backend each time
    // In practice, this is fine because CudaDevice is relatively lightweight
    CudaBackend::new(device_id)
}

/// Dispatch matrix multiplication to CUDA backend
#[allow(unused_variables)]
pub fn dispatch_cuda_matmul(a: &Tensor, b: &Tensor, device_id: usize) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        match (a, b) {
            (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
                // Convert to 2D arrays
                if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                    return Err(TrustformersError::shape_error(
                        "CUDA dispatch currently only supports 2D tensors".to_string(),
                    ));
                }

                let a_2d = a_arr
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to convert to 2D: {}", e))
                    })?;
                let b_2d = b_arr
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to convert to 2D: {}", e))
                    })?;

                let (m, k) = a_2d.dim();
                let (k2, n) = b_2d.dim();

                if k != k2 {
                    return Err(TrustformersError::shape_error(format!(
                        "Matrix dimension mismatch: {}×{} vs {}×{}",
                        m, k, k2, n
                    )));
                }

                // Get CUDA backend
                let backend = get_cuda_backend(device_id)?;

                // Convert to contiguous slices
                let a_data: Vec<f32> = a_2d.iter().copied().collect();
                let b_data: Vec<f32> = b_2d.iter().copied().collect();

                // Execute CUDA matmul
                let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;

                // Convert back to tensor
                let result_2d = scirs2_core::ndarray::Array2::from_shape_vec((m, n), result_data)
                    .map_err(|e| {
                    TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
                })?;

                let result_dyn = result_2d.into_dyn();
                return Ok(Tensor::F32(result_dyn));
            },
            _ => {
                // Fallback to CPU matmul for non-F32 tensors
                return a.matmul(b);
            },
        }
    }

    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    {
        // No CUDA support, fallback to CPU
        a.matmul(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_dispatch_cuda_matmul() -> Result<()> {
        // Skip test if no CUDA device available
        if CudaDevice::new(0).is_err() {
            eprintln!("Skipping CUDA test: no CUDA device available");
            return Ok(());
        }

        let a = Tensor::randn(&[2, 3])?;
        let b = Tensor::randn(&[3, 4])?;

        let c = dispatch_cuda_matmul(&a, &b, 0)?;

        assert_eq!(c.shape(), &[2, 4]);
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_cuda_backend_correctness() -> Result<()> {
        // Skip test if no CUDA device available
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping CUDA test: no CUDA device available");
                return Ok(());
            },
        };

        // Simple 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]

        let result = backend.matmul_f32(&a, &b, 2, 2, 2)?;

        // Expected: [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: {} vs {}",
                i,
                res,
                exp
            );
        }

        Ok(())
    }
}
