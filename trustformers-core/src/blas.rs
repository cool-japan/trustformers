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
    /// Tile extent used by the blocked GEMM kernels.
    ///
    /// This is what [`BlasOptimizer::auto_tune`] searches over, and it really
    /// changes the loop structure: see [`BlasOptimizer::sequential_gemm`].
    pub block_size: usize,
}

impl Default for BlasConfig {
    fn default() -> Self {
        Self {
            backend: BlasBackend::default(),
            num_threads: None, // Use global parallel context
            auto_tune: true,
            cache_kernels: true,
            block_size: 64,
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

    /// Auto-tune GEMM parameters by benchmarking the candidate strategies.
    ///
    /// For every workload shape, each `(block_size, threading)` candidate is
    /// timed against the real pure-Rust GEMM and the fastest is cached, with
    /// `performance_score` set to the measured throughput in GFLOP/s. Nothing
    /// is cached from a heuristic: the previous version picked a block size
    /// from three `m*k*n` thresholds and stored `performance_score: 1.0`, so
    /// the "tuned" configuration had never been measured at all.
    pub fn auto_tune(&mut self, workload_sizes: &[(usize, usize, usize)]) -> Result<()> {
        if !self.config.auto_tune {
            return Ok(());
        }

        const BLOCK_SIZES: [usize; 4] = [32, 64, 128, 256];

        for &(m, k, n) in workload_sizes {
            if m == 0 || k == 0 || n == 0 {
                return Err(TrustformersError::invalid_input(format!(
                    "cannot tune a degenerate GEMM shape {}x{}x{}",
                    m, k, n
                )));
            }

            // Deterministic operands, allocated once per shape.
            let a = Tensor::from_vec(
                (0..m * k).map(|i| ((i % 17) as f32) * 0.125 - 1.0).collect(),
                &[m, k],
            )?;
            let b = Tensor::from_vec(
                (0..k * n).map(|i| ((i % 13) as f32) * 0.0625 - 0.5).collect(),
                &[k, n],
            )?;

            // 2*m*n*k floating-point operations per GEMM.
            let flops = 2.0 * m as f64 * n as f64 * k as f64;

            let mut best: Option<CachedKernel> = None;
            for block_size in BLOCK_SIZES {
                for use_threading in [false, true] {
                    if use_threading && !self.config.use_parallel {
                        continue;
                    }

                    let previous_block = self.config.block_size;
                    let previous_parallel = self.config.use_parallel;
                    self.config.block_size = block_size;
                    self.config.use_parallel = use_threading;

                    // Warm up, then time the real kernel.
                    let warmup = self.pure_rust_gemm(&a, &b, 1.0, 0.0, None, m, k, n);
                    let start = std::time::Instant::now();
                    let measured = self.pure_rust_gemm(&a, &b, 1.0, 0.0, None, m, k, n);
                    let elapsed = start.elapsed();

                    self.config.block_size = previous_block;
                    self.config.use_parallel = previous_parallel;

                    warmup?;
                    measured?;

                    let seconds = elapsed.as_secs_f64();
                    if seconds <= 0.0 {
                        continue;
                    }
                    // GFLOP/s actually achieved by this configuration.
                    let performance_score = flops / seconds / 1e9;

                    let candidate = CachedKernel {
                        operation: BlasOperation::Gemm,
                        optimal_block_size: block_size,
                        use_threading,
                        performance_score,
                    };

                    if best
                        .as_ref()
                        .is_none_or(|current| performance_score > current.performance_score)
                    {
                        best = Some(candidate);
                    }
                }
            }

            let best = best.ok_or_else(|| {
                TrustformersError::runtime_error(format!(
                    "no GEMM configuration could be timed for shape {}x{}x{}; refusing to cache \
                     an unmeasured one",
                    m, k, n
                ))
            })?;

            self.kernel_cache.insert(format!("gemm_{}x{}x{}", m, k, n), best);
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

    /// GEMM via the SciRS2-backed tensor `matmul`.
    ///
    /// `Tensor::matmul` computes `A @ B` only, so this path can honour a plain
    /// `alpha = 1, beta = 0, c = None` GEMM and nothing else. A request that
    /// needs scaling or accumulation is routed to the pure-Rust kernel, which
    /// implements the full `alpha * A @ B + beta * C` — previously the extra
    /// arguments were silently discarded and the caller got `A @ B`.
    fn scirs2_gemm(
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
        let is_plain_matmul = alpha == 1.0 && beta == 0.0 && c.is_none();
        if is_plain_matmul {
            a.matmul(b)
        } else {
            self.pure_rust_gemm(a, b, alpha, beta, c, m, k, n)
        }
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

    /// Row-parallel blocked GEMM.
    ///
    /// Rows are split across threads and each thread runs the same blocked
    /// kernel as [`Self::sequential_gemm`], so `config.block_size` affects this
    /// path too.
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
        use scirs2_core::parallel_ops::*;

        let block_size = self.config.block_size.max(1);

        // Each output row is independent, so rows can be split freely.
        result.par_chunks_mut(n).enumerate().take(m).for_each(|(i, row_out)| {
            let row_a = &a_data[i * k..(i + 1) * k];
            for j0 in (0..n).step_by(block_size) {
                let j_end = (j0 + block_size).min(n);
                for p0 in (0..k).step_by(block_size) {
                    let p_end = (p0 + block_size).min(k);
                    for p in p0..p_end {
                        let a_value = alpha * row_a[p];
                        let row_b = &b_data[p * n..(p + 1) * n];
                        for j in j0..j_end {
                            row_out[j] += a_value * row_b[j];
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Blocked sequential GEMM: `result += alpha * A @ B`.
    ///
    /// Tiled over all three axes with `config.block_size`, so the tuner's
    /// choice of block size genuinely changes the memory access pattern.
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
        let block_size = self.config.block_size.max(1);

        for i0 in (0..m).step_by(block_size) {
            let i_end = (i0 + block_size).min(m);
            for j0 in (0..n).step_by(block_size) {
                let j_end = (j0 + block_size).min(n);
                for p0 in (0..k).step_by(block_size) {
                    let p_end = (p0 + block_size).min(k);
                    for i in i0..i_end {
                        for p in p0..p_end {
                            let a_value = alpha * a_data[i * k + p];
                            let row_b = &b_data[p * n..(p + 1) * n];
                            for j in j0..j_end {
                                result[i * n + j] += a_value * row_b[j];
                            }
                        }
                    }
                }
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
            let chunk_size = n.div_ceil(4); // 4 threads

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
            let chunk_size = n.div_ceil(4);

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
    blas_optimizer()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .gemm(a, b, 1.0, 0.0, None)
}

/// Optimized matrix-vector multiplication using global BLAS optimizer
pub fn optimized_gemv(a: &Tensor, x: &Tensor) -> Result<Tensor> {
    blas_optimizer()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .gemv(a, x, 1.0, 0.0, None)
}

/// Optimized vector dot product using global BLAS optimizer
pub fn optimized_dot(x: &Tensor, y: &Tensor) -> Result<f32> {
    blas_optimizer()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .dot(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `auto_tune` never measured anything — it picked a block
    /// size from three `m*k*n` thresholds and cached
    /// `performance_score: 1.0 // Would be measured`.
    #[test]
    fn test_auto_tune_measures_real_throughput() -> Result<()> {
        let mut optimizer = BlasOptimizer::new(BlasConfig {
            auto_tune: true,
            use_parallel: false,
            ..BlasConfig::default()
        });

        optimizer.auto_tune(&[(48, 48, 48)])?;

        let cached = optimizer
            .kernel_cache
            .get("gemm_48x48x48")
            .expect("the tuned shape must be cached");

        assert_ne!(
            cached.performance_score, 1.0,
            "the hardcoded placeholder score must be gone"
        );
        assert!(
            cached.performance_score > 0.0 && cached.performance_score.is_finite(),
            "throughput must be a real positive GFLOP/s figure, got {}",
            cached.performance_score
        );
        assert!(
            [32, 64, 128, 256].contains(&cached.optimal_block_size),
            "the cached block size must come from the searched set, got {}",
            cached.optimal_block_size
        );

        Ok(())
    }

    /// Regression test: the blocked GEMM ignored `block_size` entirely, so
    /// tuning it changed nothing. Every block size must give the same result,
    /// and must match a naive reference.
    #[test]
    fn test_blocked_gemm_is_correct_for_every_block_size() -> Result<()> {
        let (m, k, n) = (7usize, 5usize, 6usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.25 - 1.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5 - 2.0).collect();

        // Naive reference.
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                expected[i * n + j] = 2.0 * sum;
            }
        }

        for block_size in [1usize, 2, 3, 64, 512] {
            let optimizer = BlasOptimizer::new(BlasConfig {
                block_size,
                use_parallel: false,
                ..BlasConfig::default()
            });

            let mut result = vec![0.0f32; m * n];
            optimizer.sequential_gemm(&a_data, &b_data, &mut result, 2.0, m, k, n);

            for (index, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "block_size {block_size}, element {index}: {got} != {want}"
                );
            }
        }

        Ok(())
    }

    /// Regression test: `scirs2_gemm` discarded `alpha`, `beta` and `c`, so a
    /// caller asking for `alpha * A @ B + beta * C` silently got `A @ B`.
    #[test]
    fn test_scirs2_gemm_honours_alpha_beta_and_c() -> Result<()> {
        let optimizer = BlasOptimizer::new(BlasConfig {
            backend: BlasBackend::SciRS2,
            use_parallel: false,
            ..BlasConfig::default()
        });

        // A = [[1, 2], [3, 4]], B = I, C = ones.
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])?;
        let c = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2])?;

        // Plain matmul: A @ I = A.
        let plain = optimizer.scirs2_gemm(&a, &b, 1.0, 0.0, None, 2, 2, 2)?;
        assert_eq!(plain.data()?, vec![1.0, 2.0, 3.0, 4.0]);

        // 2*A@I + 10*C = 2A + 10.
        let scaled = optimizer.scirs2_gemm(&a, &b, 2.0, 10.0, Some(&c), 2, 2, 2)?;
        assert_eq!(
            scaled.data()?,
            vec![12.0, 14.0, 16.0, 18.0],
            "alpha, beta and C must all be honoured"
        );

        Ok(())
    }

    /// A degenerate shape cannot be tuned and must say so.
    #[test]
    fn test_auto_tune_rejects_degenerate_shapes() -> Result<()> {
        let mut optimizer = BlasOptimizer::new(BlasConfig {
            auto_tune: true,
            ..BlasConfig::default()
        });
        assert!(optimizer.auto_tune(&[(0, 4, 4)]).is_err());
        Ok(())
    }

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
        // Default is Accelerate as specified by #[default] attribute
        // Runtime fallback to available backends happens in BlasOptimizer
        assert_eq!(backend, BlasBackend::Accelerate);
    }

    #[test]
    fn test_blas_optimizer_creation() {
        let optimizer = BlasOptimizer::default();
        assert!(optimizer.config().auto_tune);
    }

    #[test]
    fn test_optimized_gemm() -> Result<()> {
        let mut optimizer = BlasOptimizer::default();

        let a =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor from_vec failed");
        let b =
            Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("Tensor from_vec failed");

        let result = optimizer.gemm(&a, &b, 1.0, 0.0, None).expect("operation failed in test");
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

        let a =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor from_vec failed");
        let x = Tensor::from_vec(vec![5.0, 6.0], &[2]).expect("Tensor from_vec failed");

        let result = optimizer.gemv(&a, &x, 1.0, 0.0, None).expect("operation failed in test");
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

        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Tensor from_vec failed");
        let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).expect("Tensor from_vec failed");

        let result = optimizer.dot(&x, &y).expect("operation failed in test");
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

        let x = Tensor::from_vec(vec![3.0, 4.0], &[2]).expect("Tensor from_vec failed");
        let result = optimizer.nrm2(&x).expect("operation failed in test");
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

        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Tensor from_vec failed");
        let result = optimizer.scal(2.0, &x).expect("operation failed in test");
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

        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Tensor from_vec failed");
        let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).expect("Tensor from_vec failed");

        let result = optimizer.axpy(2.0, &x, &y).expect("operation failed in test");
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
        optimizer.auto_tune(&workload_sizes).expect("operation failed in test");

        assert_eq!(optimizer.kernel_cache.len(), 3);
    }

    #[test]
    fn test_global_blas_optimizer() {
        let a =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor from_vec failed");
        let b =
            Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("Tensor from_vec failed");

        let result = optimized_gemm(&a, &b).expect("operation failed in test");
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
        let serialized = serde_json::to_string(&config).expect("JSON serialization failed");
        let deserialized: BlasConfig =
            serde_json::from_str(&serialized).expect("JSON deserialization failed");

        assert_eq!(config.backend, deserialized.backend);
        assert_eq!(config.auto_tune, deserialized.auto_tune);
    }
}
