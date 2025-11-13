//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::common::*;
use super::metalbackend_type::MetalBackend;
use super::types::BufferId;
/// Global Metal backend cache
#[cfg(all(target_os = "macos", feature = "metal"))]
static METAL_BACKEND: once_cell::sync::Lazy<std::sync::Mutex<Option<MetalBackend>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(None));
/// Get or create Metal backend instance
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn get_metal_backend() -> Result<MetalBackend> {
    let mut cache = METAL_BACKEND.lock().map_err(|_| {
        TrustformersError::hardware_error("Failed to lock Metal backend cache", "get_metal_backend")
    })?;
    if cache.is_none() {
        *cache = Some(MetalBackend::new()?);
    }
    cache
        .as_ref()
        .ok_or_else(|| {
            TrustformersError::hardware_error("Metal backend not initialized", "get_metal_backend")
        })
        .and_then(|backend| {
            Ok(MetalBackend {
                device: backend.device.clone(),
                command_queue: backend.command_queue.clone(),
                buffer_cache: Arc::clone(&backend.buffer_cache),
                matmul_pipeline: Arc::clone(&backend.matmul_pipeline),
                gelu_pipeline: Arc::clone(&backend.gelu_pipeline),
                scale_pipeline: Arc::clone(&backend.scale_pipeline),
                add_bias_pipeline: Arc::clone(&backend.add_bias_pipeline),
                layernorm_pipeline: Arc::clone(&backend.layernorm_pipeline),
                rope_pipeline: Arc::clone(&backend.rope_pipeline),
                softmax_causal_pipeline: Arc::clone(&backend.softmax_causal_pipeline),
                copy_with_offset_pipeline: Arc::clone(&backend.copy_with_offset_pipeline),
                elementwise_add_pipeline: Arc::clone(&backend.elementwise_add_pipeline),
                split_qkv_pipeline: Arc::clone(&backend.split_qkv_pipeline),
                transpose_pipeline: Arc::clone(&backend.transpose_pipeline),
                reshape_to_heads_pipeline: Arc::clone(&backend.reshape_to_heads_pipeline),
                reshape_from_heads_pipeline: Arc::clone(&backend.reshape_from_heads_pipeline),
                batched_transpose_pipeline: Arc::clone(&backend.batched_transpose_pipeline),
                batched_softmax_causal_pipeline: Arc::clone(
                    &backend.batched_softmax_causal_pipeline,
                ),
                batched_matmul_pipeline: Arc::clone(&backend.batched_matmul_pipeline),
                batched_matmul_scaled_pipeline: Arc::clone(&backend.batched_matmul_scaled_pipeline),
                batched_scaled_matmul_softmax_causal_pipeline: Arc::clone(
                    &backend.batched_scaled_matmul_softmax_causal_pipeline,
                ),
                concat_seq_dim_pipeline: Arc::clone(&backend.concat_seq_dim_pipeline),
                mps_ops: Arc::clone(&backend.mps_ops),
            })
        })
}
/// Dispatch matrix multiplication to appropriate backend based on device
#[allow(unused_variables)]
pub fn dispatch_matmul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Device::Metal(_device_id) = device {
            match (a, b) {
                (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
                    if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                        return Err(TrustformersError::shape_error(
                            "Metal dispatch currently only supports 2D tensors".to_string(),
                        ));
                    }
                    let a_2d = a_arr
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to convert to 2D: {}",
                                e
                            ))
                        })?;
                    let b_2d = b_arr
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to convert to 2D: {}",
                                e
                            ))
                        })?;
                    let (m, k) = a_2d.dim();
                    let (k2, n) = b_2d.dim();
                    if k != k2 {
                        return Err(TrustformersError::shape_error(format!(
                            "Matrix dimension mismatch: {}×{} vs {}×{}",
                            m, k, k2, n
                        )));
                    }
                    let backend = get_metal_backend()?;
                    let a_data: Vec<f32> = a_2d.iter().copied().collect();
                    let b_data: Vec<f32> = b_2d.iter().copied().collect();
                    let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;
                    let result_2d = scirs2_core::ndarray::Array2::from_shape_vec(
                        (m, n),
                        result_data,
                    )
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
                    })?;
                    let result_dyn = result_2d.into_dyn();
                    return Ok(Tensor::F32(result_dyn));
                },
                _ => {
                    return a.matmul(b);
                },
            }
        }
    }
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
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_dispatch_matmul_metal() -> Result<()> {
        let a = Tensor::randn(&[2, 3])?;
        let b = Tensor::randn(&[3, 4])?;
        let c = dispatch_matmul(&a, &b, &Device::Metal(0))?;
        assert_eq!(c.shape(), &[2, 4]);
        Ok(())
    }
    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_metal_backend_correctness() -> Result<()> {
        let backend = MetalBackend::new()?;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = backend.matmul_f32(&a, &b, 2, 2, 2)?;
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                res,
                exp
            );
        }
        Ok(())
    }
}
