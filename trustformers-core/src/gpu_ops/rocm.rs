//! ROCm GPU backend for tensor operations
//!
//! This module provides ROCm GPU acceleration for tensor operations on AMD GPUs.
//! ROCm (Radeon Open Compute) is AMD's open-source GPU computing platform.
//!
//! Features:
//! - AMD GPU support via HIP (Heterogeneous-compute Interface for Portability)
//! - CUDA-compatible API
//! - Persistent buffer caching
//! - GPU-to-GPU operations
//!
//! Note: ROCm support is currently a placeholder. Full implementation requires:
//! - HIP runtime installation
//! - Rust HIP bindings (hip-rs or similar)
//! - AMD GPU hardware

#[allow(unused_imports)]
use crate::device::Device;
use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(feature = "rocm")]
use crate::errors::TrustformersError;
#[cfg(feature = "rocm")]
use std::collections::HashMap;
#[cfg(feature = "rocm")]
use std::sync::Arc;

/// Buffer ID for persistent GPU buffers
#[cfg(feature = "rocm")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[cfg(feature = "rocm")]
impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[cfg(feature = "rocm")]
impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// ROCm GPU backend for matrix multiplication and element-wise operations
///
/// This is a placeholder implementation. Full ROCm support requires:
/// - HIP runtime libraries
/// - Rust HIP bindings
/// - AMD GPU hardware with ROCm drivers
#[cfg(feature = "rocm")]
pub struct RocmBackend {
    device_id: usize,
    // In a full implementation, this would contain:
    // - HIP device handle
    // - HIP stream handle
    // - Buffer cache
    // - Compiled kernels
}

#[cfg(feature = "rocm")]
impl RocmBackend {
    /// Create a new ROCm backend
    pub fn new(device_id: usize) -> Result<Self> {
        // Check if ROCm is available
        if !Self::is_rocm_available() {
            return Err(TrustformersError::hardware_error(
                "ROCm runtime not found. Please install ROCm toolkit.",
                "RocmBackend::new",
            ));
        }

        tracing::debug!("ROCm backend initialized on device {}", device_id);

        Ok(Self { device_id })
    }

    /// Check if a real ROCm/HIP runtime is available on the system.
    ///
    /// ROCm ships only for Linux, and a stale install path or leftover
    /// environment variable does not mean the runtime actually works - so
    /// this attempts to `dlopen` the real HIP runtime library rather than
    /// just checking that `/opt/rocm` exists or `ROCM_PATH`/`HIP_PATH` are
    /// set (which a broken or partial install can still leave behind).
    fn is_rocm_available() -> bool {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: `Library::new` only opens the shared object to read
            // its dynamic symbol table; it does not call into it. Any
            // handle obtained here is immediately dropped - this call is
            // solely a runtime availability probe, mirroring the same
            // library names `kernels/rocm_impl.rs::HipLibrary::load` binds.
            unsafe {
                libloading::Library::new("libamdhip64.so")
                    .or_else(|_| libloading::Library::new("libamdhip64.so.5"))
                    .or_else(|_| libloading::Library::new("libamdhip64.so.6"))
                    .is_ok()
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            // ROCm has no runtime for macOS/Windows; never claim
            // availability there regardless of stray env vars or paths.
            false
        }
    }

    /// Perform matrix multiplication.
    ///
    /// No HIP kernel is dispatched here (see the module docs: this crate
    /// carries no real HIP GEMM binding yet). Rather than a hand-rolled,
    /// unblocked triple loop - orders of magnitude slower than either a
    /// real GPU or the CPU BLAS path - this routes through `Tensor::matmul`,
    /// which uses OxiBLAS (`oxiblas_blas::level3::gemm` on macOS,
    /// scirs2-core's SIMD GEMM elsewhere) as its CPU fallback. `new()`
    /// already refuses to construct a `RocmBackend` unless
    /// `is_rocm_available()` found a real HIP runtime, so reaching this
    /// method at all only happens on a genuine (if not yet wired up here)
    /// ROCm-capable host.
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        tracing::debug!(
            "ROCm HIP kernel dispatch not yet wired up (device {}) - using the OxiBLAS-backed \
             CPU GEMM path instead of a hand-rolled loop",
            self.device_id
        );

        let a_tensor = Tensor::from_vec(a.to_vec(), &[m, k])?;
        let b_tensor = Tensor::from_vec(b.to_vec(), &[k, n])?;
        let result = a_tensor.matmul(&b_tensor)?;
        result.data()
    }

    /// Execute GELU activation on GPU (placeholder)
    pub fn gelu_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        tracing::debug!("ROCm GPU operations not yet implemented - using CPU fallback");

        // CPU fallback GELU implementation
        let result: Vec<f32> = input
            .iter()
            .map(|&x| {
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

        Ok(result)
    }

    /// Execute LayerNorm on GPU (placeholder)
    pub fn layernorm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        tracing::debug!("ROCm GPU operations not yet implemented - using CPU fallback");

        let total_size = seq_len * hidden_size;
        let mut result = vec![0.0f32; total_size];

        for pos in 0..seq_len {
            let offset = pos * hidden_size;

            // Compute mean
            let sum: f32 = input[offset..offset + hidden_size].iter().sum();
            let mean = sum / hidden_size as f32;

            // Compute variance
            let var_sum: f32 = input[offset..offset + hidden_size]
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum();
            let variance = var_sum / hidden_size as f32;
            let std_dev = (variance + eps).sqrt();

            // Normalize and apply affine transform
            for i in 0..hidden_size {
                let normalized = (input[offset + i] - mean) / std_dev;
                result[offset + i] = normalized * weight[i] + bias[i];
            }
        }

        Ok(result)
    }

    /// Get device information.
    ///
    /// Deliberately does not claim to be a GPU device: `matmul_f32` (and
    /// friends) run on the CPU via OxiBLAS (see their docs), not on the
    /// AMD GPU `is_rocm_available()` detected at construction time.
    pub fn device_info(&self) -> String {
        format!(
            "ROCm host {} - HIP runtime detected but kernel dispatch is not wired up yet; \
             operations run on the CPU via OxiBLAS, not on the GPU",
            self.device_id
        )
    }
}

/// Get or create ROCm backend instance
#[cfg(feature = "rocm")]
pub fn get_rocm_backend(device_id: usize) -> Result<Arc<RocmBackend>> {
    static ROCM_BACKENDS: once_cell::sync::Lazy<
        std::sync::Mutex<HashMap<usize, Arc<RocmBackend>>>,
    > = once_cell::sync::Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

    let mut cache = ROCM_BACKENDS.lock().map_err(|_| {
        TrustformersError::hardware_error("Failed to lock ROCm backend cache", "get_rocm_backend")
    })?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(device_id) {
        let backend = RocmBackend::new(device_id)?;
        e.insert(Arc::new(backend));
    }

    cache.get(&device_id).cloned().ok_or_else(|| {
        TrustformersError::hardware_error("ROCm backend not found", "get_rocm_backend")
    })
}

/// Dispatch matrix multiplication to ROCm backend
#[allow(unused_variables)]
pub fn dispatch_rocm_matmul(a: &Tensor, b: &Tensor, device_id: usize) -> Result<Tensor> {
    #[cfg(feature = "rocm")]
    {
        match (a, b) {
            (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
                if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                    return Err(TrustformersError::shape_error(
                        "ROCm dispatch currently only supports 2D tensors".to_string(),
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

                let backend = get_rocm_backend(device_id)?;

                let a_data: Vec<f32> = a_2d.iter().copied().collect();
                let b_data: Vec<f32> = b_2d.iter().copied().collect();

                let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;

                let result_2d = scirs2_core::ndarray::Array2::from_shape_vec((m, n), result_data)
                    .map_err(|e| {
                    TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
                })?;

                let result_dyn = result_2d.into_dyn();
                Ok(Tensor::F32(result_dyn))
            },
            _ => a.matmul(b),
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        // No ROCm support, fallback to CPU
        a.matmul(b)
    }
}

#[cfg(all(test, feature = "rocm"))]
impl RocmBackend {
    /// Test-only constructor that bypasses `is_rocm_available()`.
    /// `RocmBackend` holds no live device resource (just `device_id`), so
    /// this is safe and lets `matmul_f32`/`device_info` be exercised for
    /// correctness on hosts with no real HIP runtime (e.g. CI on macOS),
    /// independent of whether hardware *detection* itself succeeds.
    fn new_for_test(device_id: usize) -> Self {
        Self { device_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "rocm")]
    fn test_rocm_availability() {
        let available = RocmBackend::is_rocm_available();
        println!("ROCm available: {}", available);
    }

    /// Regression test: before this fix, availability was based on
    /// `/opt/rocm` existing or `ROCM_PATH`/`HIP_PATH` being set - true even
    /// on a non-Linux host with a stray directory or leftover env var and
    /// no real HIP runtime at all. ROCm ships only for Linux, so this must
    /// be unconditionally false elsewhere.
    #[test]
    #[cfg(all(feature = "rocm", not(target_os = "linux")))]
    fn test_is_rocm_available_false_on_non_linux() {
        assert!(!RocmBackend::is_rocm_available());
    }

    /// Regression test: before this fix, `matmul_f32` computed via a
    /// hand-rolled `O(m*n*k)` triple loop. This is numerically fine but is
    /// replaced with the OxiBLAS-backed `Tensor::matmul` path; verify the
    /// rewrite still produces correct results for a *non-square* matmul
    /// (the shape most likely to expose a transpose/stride mix-up when
    /// switching implementations).
    #[test]
    #[cfg(feature = "rocm")]
    fn test_matmul_f32_rectangular_matches_hand_computed_reference() {
        let backend = RocmBackend::new_for_test(0);
        // A: 2x3, B: 3x2 -> C: 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let result = backend.matmul_f32(&a, &b, 2, 3, 2).expect("matmul_f32 should succeed");

        // row0 = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row1 = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let expected = [58.0, 64.0, 139.0, 154.0];
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-3,
                "mismatch at index {}: {} vs {}",
                i,
                res,
                exp
            );
        }
    }

    /// Regression test: before this fix, `device_info` unconditionally
    /// read "ROCm Device N (placeholder - HIP bindings required)" - easily
    /// misread as "a GPU is in use". It must now make clear execution
    /// happens on the CPU.
    #[test]
    #[cfg(feature = "rocm")]
    fn test_device_info_does_not_claim_gpu_execution() {
        let backend = RocmBackend::new_for_test(0);
        let info = backend.device_info();
        assert!(
            info.contains("CPU"),
            "device_info must make clear execution is on the CPU, got: {info}"
        );
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_rocm_backend() -> Result<()> {
        match RocmBackend::new(0) {
            Ok(backend) => {
                println!("ROCm backend: {}", backend.device_info());
                Ok(())
            },
            Err(_) => {
                tracing::debug!("Skipping ROCm test: not available");
                Ok(())
            },
        }
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_rocm_matmul_fallback() -> Result<()> {
        let backend = match RocmBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                tracing::debug!("Skipping ROCm test: not available");
                return Ok(());
            },
        };

        // Test CPU fallback
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = backend.matmul_f32(&a, &b, 2, 2, 2)?;

        // Expected: [[19, 22], [43, 50]]
        let expected = [19.0, 22.0, 43.0, 50.0];

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
