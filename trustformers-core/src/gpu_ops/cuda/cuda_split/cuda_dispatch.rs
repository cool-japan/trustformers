//! CUDA dispatch functions for tensor operations

use crate::device::Device;
use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use crate::errors::TrustformersError;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::sync::Arc;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use super::cuda_backend::*;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub fn get_cuda_backend(device_id: usize) -> Result<Arc<CudaBackend>> {
    let mut cache = CUDA_BACKENDS.lock().map_err(|_| {
        TrustformersError::hardware_error("Failed to lock CUDA backend cache", "get_cuda_backend")
    })?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(device_id) {
        let backend = CudaBackend::new(device_id)?;
        e.insert(Arc::new(backend));
    }

    cache.get(&device_id).cloned().ok_or_else(|| {
        TrustformersError::hardware_error("CUDA backend not found", "get_cuda_backend")
    })
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
                Ok(Tensor::F32(result_dyn))
            },
            _ => {
                // Fallback to CPU matmul for non-F32 tensors
                a.matmul(b)
            },
        }
    }

    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    {
        // No CUDA support, fallback to CPU
        a.matmul(b)
    }
}

/// Dispatch matrix multiplication to appropriate backend based on device
#[allow(unused_variables)]
pub fn dispatch_matmul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        if let Device::CUDA(device_id) = device {
            return dispatch_cuda_matmul(a, b, *device_id);
        }
    }

    // Default: CPU matmul
    a.matmul(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_matmul_cpu() -> Result<()> {
        let a = Tensor::randn(&[2, 3])?;
        let b = Tensor::randn(&[3, 4])?;

        let c = dispatch_matmul(&a, &b, &Device::CPU)?;

        assert_eq!(c.shape(), &[2, 4]);
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_dispatch_cuda_matmul() -> Result<()> {
        use cudarc::driver::CudaContext;

        // Skip test if no CUDA device available
        if CudaContext::new(0).is_err() {
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
        let expected = [19.0_f32, 22.0_f32, 43.0_f32, 50.0_f32];

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

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_cuda_gelu() -> Result<()> {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping CUDA test: no CUDA device available");
                return Ok(());
            },
        };

        // Test GELU on various inputs
        // Note: GELU has a local minimum around x ≈ -0.8, so it's NOT monotonic over all ranges
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = backend.gelu_f32(&input)?;

        // GELU(0) ≈ 0
        assert!((result[2]).abs() < 0.01, "GELU(0) should be ~0");

        // Check approximate expected values for GELU
        // GELU(-2) ≈ -0.045, GELU(-1) ≈ -0.159, GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
        let expected = [-0.045, -0.159, 0.0, 0.841, 1.955];
        for i in 0..input.len() {
            assert!(
                (result[i] - expected[i]).abs() < 0.05,
                "GELU({}) = {} but expected ~{}",
                input[i],
                result[i],
                expected[i]
            );
        }

        // GELU is monotonic increasing only for x >= ~-0.5
        // Test monotonicity in the positive region
        let positive_input = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let positive_result = backend.gelu_f32(&positive_input)?;
        for i in 0..positive_result.len() - 1 {
            assert!(
                positive_result[i] <= positive_result[i + 1] + 1e-5,
                "GELU should be monotonic in positive region"
            );
        }

        Ok(())
    }

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_cuda_layernorm() -> Result<()> {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping CUDA test: no CUDA device available");
                return Ok(());
            },
        };

        // Simple test: 2 sequences, 4 features each
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let result = backend.layernorm_f32(&input, &weight, &bias, 2, 4, 1e-5)?;

        assert_eq!(result.len(), 8);

        // Each row should have mean ≈ 0 and std ≈ 1
        for row in 0..2 {
            let row_data = &result[row * 4..(row + 1) * 4];
            let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);
        }

        Ok(())
    }

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_cuda_buffer_cache() -> Result<()> {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping CUDA test: no CUDA device available");
                return Ok(());
            },
        };

        // Create a persistent buffer
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buffer_id = backend.create_persistent_buffer(&data)?;

        // Check cache size
        assert_eq!(backend.buffer_cache_size()?, 1);

        // Retrieve buffer
        let retrieved = backend.buffer_to_cpu(&buffer_id, 4)?;
        assert_eq!(retrieved, data);

        // Remove buffer
        backend.remove_persistent_buffer(&buffer_id)?;
        assert_eq!(backend.buffer_cache_size()?, 0);

        Ok(())
    }

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_cuda_gpu_to_gpu_operations() -> Result<()> {
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => {
                eprintln!("Skipping CUDA test: no CUDA device available");
                return Ok(());
            },
        };

        // Test GPU-to-GPU GELU
        let input_data = vec![0.0, 1.0, 2.0, 3.0];
        let input_id = backend.create_persistent_buffer(&input_data)?;

        let output_id = backend.gelu_gpu_to_gpu(&input_id, 4)?;

        // Retrieve result
        let result = backend.buffer_to_cpu(&output_id, 4)?;
        assert_eq!(result.len(), 4);

        // GELU(0) ≈ 0
        assert!(result[0].abs() < 0.01);

        // Clean up
        backend.clear_buffer_cache()?;

        Ok(())
    }
}
