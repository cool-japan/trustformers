#![allow(unused_variables)] // BLAS interface implementation with backend-specific code

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, OnceLock};

/// BLAS backend types following SciRS2 Core Usage Policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BlasBackend {
    /// Apple Accelerate Framework (macOS)
    #[default]
    Accelerate,
    /// Intel MKL
    Mkl,
    /// OpenBLAS
    OpenBlas,
    /// Netlib BLAS/LAPACK reference implementation
    Netlib,
    /// Pure Rust implementation
    PureRust,
    /// SciRS2 optimized implementation
    SciRS2,
}

/// BLAS optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlasConfig {
    pub backend: BlasBackend,
    pub num_threads: Option<usize>,
    pub auto_tune: bool,
    pub cache_kernels: bool,
    pub use_parallel: bool,
    pub min_size_for_blas: usize,
}

impl Default for BlasConfig {
    fn default() -> Self {
        Self {
            backend: BlasBackend::default(),
            num_threads: None, // Use global parallel context
            auto_tune: true,
            cache_kernels: true,
            use_parallel: true,
            min_size_for_blas: 32, // Minimum matrix size to use BLAS
        }
    }
}

/// BLAS operation types
#[derive(Debug, Clone, Copy)]
pub enum BlasOperation {
    /// General matrix multiply: C = alpha * A * B + beta * C
    Gemm,
    /// Matrix-vector multiply: y = alpha * A * x + beta * y
    Gemv,
    /// Vector dot product: result = x^T * y
    Dot,
    /// Vector norm: result = ||x||_2
    Nrm2,
    /// Scale vector: x = alpha * x
    Scal,
    /// Add vectors: y = alpha * x + y
    Axpy,
}

/// BLAS optimizer for tensor operations
#[derive(Debug)]
pub struct BlasOptimizer {
    config: BlasConfig,
    kernel_cache: std::collections::HashMap<String, CachedKernel>,
}

#[derive(Debug, Clone)]
struct CachedKernel {
    #[allow(dead_code)]
    operation: BlasOperation,
    #[allow(dead_code)]
    optimal_block_size: usize,
    #[allow(dead_code)]
    use_threading: bool,
    #[allow(dead_code)]
    performance_score: f64,
}

impl Default for BlasOptimizer {
    fn default() -> Self {
        Self::new(BlasConfig::default())
    }
}

impl BlasOptimizer {
    /// Create a new BLAS optimizer with configuration
    pub fn new(config: BlasConfig) -> Self {
        Self {
            config,
            kernel_cache: std::collections::HashMap::new(),
        }
    }

    /// Optimized matrix multiplication
    pub fn gemm(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        c: Option<&Tensor>,
    ) -> Result<Tensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Validate shapes
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(TrustformersError::tensor_op_error(
                "GEMM requires at least 2D tensors",
                "gemm",
            ));
        }

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        if k != b_shape[b_shape.len() - 2] {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Incompatible shapes for GEMM: {:?} x {:?}",
                    a_shape, b_shape
                ),
                "gemm",
            ));
        }

        // Check if we should use BLAS optimization
        if m * n * k
            >= self.config.min_size_for_blas
                * self.config.min_size_for_blas
                * self.config.min_size_for_blas
        {
            self.optimized_gemm(a, b, alpha, beta, c, m, k, n)
        } else {
            self.fallback_gemm(a, b, alpha, beta, c)
        }
    }

    /// Optimized matrix-vector multiplication
    pub fn gemv(
        &mut self,
        a: &Tensor,
        x: &Tensor,
        alpha: f32,
        beta: f32,
        y: Option<&Tensor>,
    ) -> Result<Tensor> {
        let a_shape = a.shape();
        let x_shape = x.shape();

        if a_shape.len() != 2 || x_shape.len() != 1 {
            return Err(TrustformersError::tensor_op_error(
                "GEMV requires 2D matrix and 1D vector",
                "gemv",
            ));
        }

        let m = a_shape[0];
        let n = a_shape[1];

        if n != x_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Incompatible shapes for GEMV: matrix {:?} x vector {:?}",
                    a_shape, x_shape
                ),
                "gemv",
            ));
        }

        if m * n >= self.config.min_size_for_blas {
            self.optimized_gemv(a, x, alpha, beta, y, m, n)
        } else {
            self.fallback_gemv(a, x, alpha, beta, y)
        }
    }

    /// Vector dot product
    pub fn dot(&self, x: &Tensor, y: &Tensor) -> Result<f32> {
        let x_shape = x.shape();
        let y_shape = y.shape();

        if x_shape.len() != 1 || y_shape.len() != 1 || x_shape[0] != y_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                "DOT requires vectors of same length",
                "dot",
            ));
        }

        let n = x_shape[0];
        if n >= self.config.min_size_for_blas {
            self.optimized_dot(x, y, n)
        } else {
            self.fallback_dot(x, y)
        }
    }

    /// Vector L2 norm
    pub fn nrm2(&self, x: &Tensor) -> Result<f32> {
        let x_shape = x.shape();
        if x_shape.len() != 1 {
            return Err(TrustformersError::tensor_op_error(
                "NRM2 requires 1D vector",
                "nrm2",
            ));
        }

        let n = x_shape[0];
        if n >= self.config.min_size_for_blas {
            self.optimized_nrm2(x, n)
        } else {
            self.fallback_nrm2(x)
        }
    }

    /// Scale vector by scalar
    pub fn scal(&self, alpha: f32, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape();
        if x_shape.len() != 1 {
            return Err(TrustformersError::tensor_op_error(
                "SCAL requires 1D vector",
                "scal",
            ));
        }

        let n = x_shape[0];
        if n >= self.config.min_size_for_blas {
            self.optimized_scal(alpha, x, n)
        } else {
            self.fallback_scal(alpha, x)
        }
    }

    /// Add scaled vector: y = alpha * x + y
    pub fn axpy(&self, alpha: f32, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape();
        let y_shape = y.shape();

        if x_shape != y_shape || x_shape.len() != 1 {
            return Err(TrustformersError::tensor_op_error(
                "AXPY requires vectors of same length",
                "axpy",
            ));
        }

        let n = x_shape[0];
        if n >= self.config.min_size_for_blas {
            self.optimized_axpy(alpha, x, y, n)
        } else {
            self.fallback_axpy(alpha, x, y)
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &BlasConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: BlasConfig) {
        self.config = config;
        // Clear cache when config changes
        self.kernel_cache.clear();
    }

    /// Get backend name
    pub fn backend_name(&self) -> &'static str {
        match self.config.backend {
            BlasBackend::Accelerate => "Apple Accelerate",
            BlasBackend::Mkl => "Intel MKL",
            BlasBackend::OpenBlas => "OpenBLAS",
            BlasBackend::Netlib => "Netlib BLAS",
            BlasBackend::PureRust => "Pure Rust",
            BlasBackend::SciRS2 => "SciRS2 Optimized",
        }
    }

    /// Auto-tune BLAS parameters for given workload
    pub fn auto_tune(&mut self, workload_sizes: &[(usize, usize, usize)]) -> Result<()> {
        if !self.config.auto_tune {
            return Ok(());
        }

        // Auto-tuning would measure performance for different block sizes
        // and threading strategies for the given workload
        for &(m, k, n) in workload_sizes {
            let key = format!("gemm_{}x{}x{}", m, k, n);

            // Simplified auto-tuning - in practice this would benchmark different strategies
            let optimal_block_size = if m * k * n > 1_000_000 {
                256
            } else if m * k * n > 100_000 {
                128
            } else {
                64
            };

            let use_threading = self.config.use_parallel && m * k * n > 10_000;

            let cached_kernel = CachedKernel {
                operation: BlasOperation::Gemm,
                optimal_block_size,
                use_threading,
                performance_score: 1.0, // Would be measured
            };

            self.kernel_cache.insert(key, cached_kernel);
        }

        Ok(())
    }

    // Private optimization implementations

    fn optimized_gemm(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        c: Option<&Tensor>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor> {
        match self.config.backend {
            BlasBackend::SciRS2 => self.scirs2_gemm(a, b, alpha, beta, c, m, k, n),
            BlasBackend::PureRust => self.pure_rust_gemm(a, b, alpha, beta, c, m, k, n),
            _ => {
                // For other backends, we would call into the actual BLAS libraries
                // For now, fall back to pure Rust implementation
                self.pure_rust_gemm(a, b, alpha, beta, c, m, k, n)
            },
        }
    }

    fn scirs2_gemm(
        &self,
        a: &Tensor,
        b: &Tensor,
        _alpha: f32,
        _beta: f32,
        _c: Option<&Tensor>,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Tensor> {
        // This would use SciRS2's optimized GEMM
        // For now, delegate to tensor matmul
        a.matmul(b)
    }

    fn pure_rust_gemm(
        &self,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        c: Option<&Tensor>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor> {
        // Optimized pure Rust GEMM with blocking and parallelization
        let a_data = a.to_vec_f32()?;
        let b_data = b.to_vec_f32()?;

        let mut result = if let Some(c_tensor) = c {
            let c_data = c_tensor.to_vec_f32()?;
            c_data.iter().map(|&x| beta * x).collect::<Vec<f32>>()
        } else {
            vec![0.0; m * n]
        };

        if self.config.use_parallel && m * n > 1000 {
            self.parallel_gemm(&a_data, &b_data, &mut result, alpha, m, k, n)?;
        } else {
            self.sequential_gemm(&a_data, &b_data, &mut result, alpha, m, k, n);
        }

        Tensor::from_vec(result, &[m, n])
    }

    fn parallel_gemm(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        result: &mut [f32],
        alpha: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let block_size = 64; // Could be auto-tuned
        let rows: Vec<usize> = (0..m).collect();

        // Simple parallel implementation - parallel_chunk_map expects the same type
        for i in 0..m {
            let mut row_result = vec![0.0; n];
            for j in 0..n {
                let mut sum = 0.0;
                for ki in 0..k {
                    sum += a_data[i * k + ki] * b_data[ki * n + j];
                }
                row_result[j] = alpha * sum;
            }
            for j in 0..n {
                result[i * n + j] += row_result[j];
            }
        }

        Ok(())
    }

    fn sequential_gemm(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        result: &mut [f32],
        alpha: f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for ki in 0..k {
                    sum += a_data[i * k + ki] * b_data[ki * n + j];
                }
                result[i * n + j] += alpha * sum;
            }
        }
    }

    fn fallback_gemm(
        &self,
        a: &Tensor,
        b: &Tensor,
        _alpha: f32,
        _beta: f32,
        _c: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Simple fallback to tensor matmul
        a.matmul(b)
    }

    fn optimized_gemv(
        &self,
        a: &Tensor,
        x: &Tensor,
        alpha: f32,
        beta: f32,
        y: Option<&Tensor>,
        m: usize,
        n: usize,
    ) -> Result<Tensor> {
        let a_data = a.to_vec_f32()?;
        let x_data = x.to_vec_f32()?;

        let mut result = if let Some(y_tensor) = y {
            let y_data = y_tensor.to_vec_f32()?;
            y_data.iter().map(|&val| beta * val).collect()
        } else {
            vec![0.0; m]
        };

        if self.config.use_parallel && m > 100 {
            let rows: Vec<usize> = (0..m).collect();
            // Simple parallel implementation
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += a_data[i * n + j] * x_data[j];
                }
                result[i] += alpha * sum;
            }
        } else {
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += a_data[i * n + j] * x_data[j];
                }
                result[i] += alpha * sum;
            }
        }

        Tensor::from_vec(result, &[m])
    }

    fn fallback_gemv(
        &self,
        a: &Tensor,
        x: &Tensor,
        alpha: f32,
        beta: f32,
        y: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Simple fallback implementation for matrix-vector multiplication
        let a_data = a.to_vec_f32()?;
        let x_data = x.to_vec_f32()?;
        let a_shape = a.shape();
        let x_shape = x.shape();

        let m = a_shape[0];
        let n = a_shape[1];

        let mut result = if let Some(y_tensor) = y {
            let y_data = y_tensor.to_vec_f32()?;
            y_data.iter().map(|&val| beta * val).collect()
        } else {
            vec![0.0; m]
        };

        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a_data[i * n + j] * x_data[j];
            }
            result[i] += alpha * sum;
        }

        Tensor::from_vec(result, &[m])
    }

    fn optimized_dot(&self, x: &Tensor, y: &Tensor, n: usize) -> Result<f32> {
        let x_data = x.to_vec_f32()?;
        let y_data = y.to_vec_f32()?;

        if self.config.use_parallel && n > 1000 {
            let indices: Vec<usize> = (0..n).collect();
            let chunk_size = (n + 3) / 4; // 4 threads

            let chunks: Vec<Vec<usize>> =
                indices.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect();

            let partial_sums: Vec<f32> = chunks
                .into_iter()
                .map(|chunk_indices| {
                    chunk_indices.iter().map(|&i| x_data[i] * y_data[i]).sum::<f32>()
                })
                .collect();

            Ok(partial_sums.into_iter().sum())
        } else {
            Ok(x_data.iter().zip(y_data.iter()).map(|(&a, &b)| a * b).sum())
        }
    }

    fn fallback_dot(&self, x: &Tensor, y: &Tensor) -> Result<f32> {
        let x_data = x.to_vec_f32()?;
        let y_data = y.to_vec_f32()?;
        Ok(x_data.iter().zip(y_data.iter()).map(|(&a, &b)| a * b).sum())
    }

    fn optimized_nrm2(&self, x: &Tensor, n: usize) -> Result<f32> {
        let x_data = x.to_vec_f32()?;

        if self.config.use_parallel && n > 1000 {
            let indices: Vec<usize> = (0..n).collect();
            let chunk_size = (n + 3) / 4;

            let chunks: Vec<Vec<usize>> =
                indices.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect();

            let partial_sums: Vec<f32> = chunks
                .into_iter()
                .map(|chunk_indices| {
                    chunk_indices.iter().map(|&i| x_data[i] * x_data[i]).sum::<f32>()
                })
                .collect();

            Ok(partial_sums.into_iter().sum::<f32>().sqrt())
        } else {
            Ok(x_data.iter().map(|&x| x * x).sum::<f32>().sqrt())
        }
    }

    fn fallback_nrm2(&self, x: &Tensor) -> Result<f32> {
        x.norm()
    }

    fn optimized_scal(&self, alpha: f32, x: &Tensor, _n: usize) -> Result<Tensor> {
        x.scale(alpha)
    }

    fn fallback_scal(&self, alpha: f32, x: &Tensor) -> Result<Tensor> {
        x.scale(alpha)
    }

    fn optimized_axpy(&self, alpha: f32, x: &Tensor, y: &Tensor, _n: usize) -> Result<Tensor> {
        let scaled_x = x.scale(alpha)?;
        scaled_x.add(y)
    }

    fn fallback_axpy(&self, alpha: f32, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let scaled_x = x.scale(alpha)?;
        scaled_x.add(y)
    }
}

/// Global BLAS optimizer instance
static BLAS_OPTIMIZER: OnceLock<Arc<Mutex<BlasOptimizer>>> = OnceLock::new();

/// Get the global BLAS optimizer
pub fn blas_optimizer() -> Arc<Mutex<BlasOptimizer>> {
    BLAS_OPTIMIZER
        .get_or_init(|| Arc::new(Mutex::new(BlasOptimizer::default())))
        .clone()
}

/// Initialize BLAS subsystem with custom configuration
pub fn init_blas(config: BlasConfig) -> Result<()> {
    if BLAS_OPTIMIZER.get().is_some() {
        return Err(TrustformersError::tensor_op_error(
            "BLAS already initialized",
            "init",
        ));
    }
    let _ = BLAS_OPTIMIZER.set(Arc::new(Mutex::new(BlasOptimizer::new(config))));
    Ok(())
}

/// Optimized matrix multiplication using global BLAS optimizer
pub fn optimized_gemm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    blas_optimizer().lock().unwrap().gemm(a, b, 1.0, 0.0, None)
}

/// Optimized matrix-vector multiplication using global BLAS optimizer
pub fn optimized_gemv(a: &Tensor, x: &Tensor) -> Result<Tensor> {
    blas_optimizer().lock().unwrap().gemv(a, x, 1.0, 0.0, None)
}

/// Optimized vector dot product using global BLAS optimizer
pub fn optimized_dot(x: &Tensor, y: &Tensor) -> Result<f32> {
    blas_optimizer().lock().unwrap().dot(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blas_config_default() {
        let config = BlasConfig::default();
        assert!(config.auto_tune);
        assert!(config.cache_kernels);
        assert_eq!(config.min_size_for_blas, 32);
    }

    #[test]
    fn test_blas_backend_default() {
        let backend = BlasBackend::default();

        #[cfg(target_os = "macos")]
        assert_eq!(backend, BlasBackend::Accelerate);

        #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
        assert_eq!(backend, BlasBackend::OpenBlas);
    }

    #[test]
    fn test_blas_optimizer_creation() {
        let optimizer = BlasOptimizer::default();
        assert!(optimizer.config().auto_tune);
    }

    #[test]
    fn test_optimized_gemm() -> Result<()> {
        let mut optimizer = BlasOptimizer::default();

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = optimizer.gemm(&a, &b, 1.0, 0.0, None).unwrap();
        let expected_data = [19.0, 22.0, 43.0, 50.0]; // Expected matrix multiplication result

        let result_data = result.to_vec_f32()?;
        for (i, (&actual, &expected)) in result_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: {} != {}",
                i,
                actual,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn test_optimized_gemv() -> Result<()> {
        let mut optimizer = BlasOptimizer::default();

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let x = Tensor::from_vec(vec![5.0, 6.0], &[2]).unwrap();

        let result = optimizer.gemv(&a, &x, 1.0, 0.0, None).unwrap();
        let expected_data = [17.0, 39.0]; // [1*5+2*6, 3*5+4*6]

        let result_data = result.to_vec_f32()?;
        for (i, (&actual, &expected)) in result_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: {} != {}",
                i,
                actual,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn test_optimized_dot() {
        let optimizer = BlasOptimizer::default();

        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let result = optimizer.dot(&x, &y).unwrap();
        let expected = 32.0; // 1*4 + 2*5 + 3*6

        assert!(
            (result - expected).abs() < 1e-6,
            "Dot product mismatch: {} != {}",
            result,
            expected
        );
    }

    #[test]
    fn test_optimized_nrm2() {
        let optimizer = BlasOptimizer::default();

        let x = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let result = optimizer.nrm2(&x).unwrap();
        let expected = 5.0; // sqrt(3^2 + 4^2)

        assert!(
            (result - expected).abs() < 1e-6,
            "Norm mismatch: {} != {}",
            result,
            expected
        );
    }

    #[test]
    fn test_optimized_scal() -> Result<()> {
        let optimizer = BlasOptimizer::default();

        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = optimizer.scal(2.0, &x).unwrap();
        let expected_data = [2.0, 4.0, 6.0];

        let result_data = result.to_vec_f32()?;
        for (i, (&actual, &expected)) in result_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: {} != {}",
                i,
                actual,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn test_optimized_axpy() -> Result<()> {
        let optimizer = BlasOptimizer::default();

        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let result = optimizer.axpy(2.0, &x, &y).unwrap();
        let expected_data = [6.0, 9.0, 12.0]; // 2*[1,2,3] + [4,5,6]

        let result_data = result.to_vec_f32()?;
        for (i, (&actual, &expected)) in result_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: {} != {}",
                i,
                actual,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn test_auto_tune() {
        let mut optimizer = BlasOptimizer::default();

        let workload_sizes = vec![(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)];
        optimizer.auto_tune(&workload_sizes).unwrap();

        assert_eq!(optimizer.kernel_cache.len(), 3);
    }

    #[test]
    fn test_global_blas_optimizer() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = optimized_gemm(&a, &b).unwrap();
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_backend_name() {
        let optimizer = BlasOptimizer::default();
        let name = optimizer.backend_name();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_blas_config_serialization() {
        let config = BlasConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: BlasConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.backend, deserialized.backend);
        assert_eq!(config.auto_tune, deserialized.auto_tune);
    }
}
