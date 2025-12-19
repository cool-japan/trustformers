//! Automatic kernel tuning for hardware adaptation
//!
//! This module provides automatic performance tuning for kernel operations across
//! different hardware backends. It profiles kernel execution times and adaptively
//! selects optimal parameters (block sizes, thread counts, memory layouts) for the
//! specific hardware being used.
//!
//! # Features
//!
//! - **Auto-tuning:** Automatic parameter selection through benchmarking
//! - **Hardware Detection:** Platform capability detection and profiling
//! - **Caching:** Persistent tuning results for faster subsequent runs
//! - **Multi-Backend:** Support for CUDA, ROCm, Metal, CPU, and more
//! - **Adaptive:** Dynamic adjustment based on tensor sizes and operations
//!
//! # Examples
//!
//! ```rust
//! use trustformers_core::kernel_tuning::{KernelTuner, TuningConfig, Operation};
//! use trustformers_core::tensor::Tensor;
//!
//! // Create tuner with default configuration
//! let mut tuner = KernelTuner::new(TuningConfig::default())?;
//!
//! // Auto-tune matrix multiplication parameters
//! let a = Tensor::randn(&[1024, 768])?;
//! let b = Tensor::randn(&[768, 512])?;
//!
//! let params = tuner.tune_matmul(&a, &b)?;
//! println!("Optimal block size: {:?}", params.block_size);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::errors::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Kernel operation types for tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation {
    /// Matrix multiplication (GEMM)
    MatMul,
    /// Convolution operation
    Convolution,
    /// Softmax activation
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// Attention computation
    Attention,
    /// Element-wise operations
    ElementWise,
    /// Reduction operations
    Reduction,
    /// Transpose/permute
    Transpose,
}

/// Hardware backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Backend {
    /// CPU backend
    CPU,
    /// NVIDIA CUDA
    CUDA,
    /// AMD ROCm/HIP
    ROCm,
    /// Apple Metal
    Metal,
    /// Vulkan Compute
    Vulkan,
    /// Intel oneAPI
    OneAPI,
    /// Google TPU
    TPU,
}

/// Platform characteristics for tuning decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Backend type
    pub backend: Backend,

    /// Device name (e.g., "NVIDIA RTX 4090", "Apple M3 Max")
    pub device_name: String,

    /// Number of compute units (SMs, CUs, cores)
    pub compute_units: usize,

    /// Total memory in bytes
    pub total_memory: usize,

    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f32,

    /// Peak compute performance in TFLOPS
    pub peak_tflops: f32,

    /// Cache sizes (L1, L2, L3) in bytes
    pub cache_sizes: Vec<usize>,

    /// Warp/wavefront size
    pub warp_size: usize,

    /// Maximum threads per block/workgroup
    pub max_threads_per_block: usize,
}

impl PlatformInfo {
    /// Detect current platform characteristics
    pub fn detect() -> Result<Self> {
        // This would use actual hardware detection APIs
        // Simplified implementation for now
        Ok(Self {
            backend: Backend::CPU,
            device_name: "Generic CPU".to_string(),
            compute_units: num_cpus::get(),
            total_memory: 16 * 1024 * 1024 * 1024, // 16GB default
            memory_bandwidth: 50.0,                // GB/s
            peak_tflops: 1.0,
            cache_sizes: vec![32768, 262144, 8388608], // L1: 32KB, L2: 256KB, L3: 8MB
            warp_size: 1,
            max_threads_per_block: 256,
        })
    }

    /// Create platform info for CUDA device
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: usize) -> Result<Self> {
        // Would query actual CUDA device properties
        Ok(Self {
            backend: Backend::CUDA,
            device_name: format!("CUDA Device {}", device_id),
            compute_units: 128,
            total_memory: 24 * 1024 * 1024 * 1024,
            memory_bandwidth: 900.0,
            peak_tflops: 82.0,
            cache_sizes: vec![128 * 1024, 40 * 1024 * 1024], // L1: 128KB, L2: 40MB
            warp_size: 32,
            max_threads_per_block: 1024,
        })
    }

    /// Get optimal block size based on hardware characteristics
    pub fn suggested_block_size(&self, operation: Operation) -> (usize, usize, usize) {
        match self.backend {
            Backend::CUDA => {
                // CUDA-specific block sizes
                match operation {
                    Operation::MatMul => (16, 16, 1),
                    Operation::Convolution => (16, 16, 1),
                    Operation::Softmax => (256, 1, 1),
                    Operation::LayerNorm => (256, 1, 1),
                    Operation::Attention => (64, 1, 1),
                    Operation::ElementWise => (256, 1, 1),
                    Operation::Reduction => (256, 1, 1),
                    Operation::Transpose => (32, 8, 1),
                }
            },
            Backend::CPU => {
                // CPU tile sizes (for blocked algorithms)
                match operation {
                    Operation::MatMul => (64, 64, 64),
                    _ => (32, 32, 1),
                }
            },
            _ => (16, 16, 1), // Conservative default
        }
    }
}

/// Tuned kernel parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelParams {
    /// Operation type
    pub operation: Operation,

    /// Block/tile size (x, y, z)
    pub block_size: (usize, usize, usize),

    /// Thread count per block
    pub threads_per_block: usize,

    /// Use shared/local memory
    pub use_shared_memory: bool,

    /// Unroll factor for loops
    pub unroll_factor: usize,

    /// Vectorization width (1, 2, 4, 8, 16)
    pub vector_width: usize,

    /// Grid dimensions
    pub grid_size: (usize, usize, usize),

    /// Estimated execution time in microseconds
    pub estimated_time_us: f64,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            operation: Operation::ElementWise,
            block_size: (16, 16, 1),
            threads_per_block: 256,
            use_shared_memory: true,
            unroll_factor: 4,
            vector_width: 4,
            grid_size: (1, 1, 1),
            estimated_time_us: 0.0,
        }
    }
}

/// Tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningConfig {
    /// Enable auto-tuning (vs. using cached results)
    pub enable_tuning: bool,

    /// Number of warmup iterations
    pub warmup_iterations: usize,

    /// Number of benchmark iterations
    pub benchmark_iterations: usize,

    /// Cache directory for tuning results
    pub cache_dir: Option<PathBuf>,

    /// Maximum tuning time per kernel in seconds
    pub max_tuning_time_secs: f32,

    /// Minimum performance improvement threshold (fraction)
    pub min_improvement_threshold: f32,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            enable_tuning: true,
            warmup_iterations: 3,
            benchmark_iterations: 10,
            cache_dir: Some(PathBuf::from(".kernel_cache")),
            max_tuning_time_secs: 10.0,
            min_improvement_threshold: 0.05, // 5% improvement
        }
    }
}

/// Tuning result for a specific configuration
#[derive(Debug, Clone)]
struct TuningResult {
    params: KernelParams,
    mean_time: Duration,
    #[allow(dead_code)]
    std_dev: f64,
}

/// Cache key for tuning results
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct CacheKey {
    operation: Operation,
    backend: Backend,
    device_name: String,
    input_shape: Vec<usize>,
}

/// Automatic kernel tuner
pub struct KernelTuner {
    /// Tuning configuration
    config: TuningConfig,

    /// Platform information
    platform: PlatformInfo,

    /// Cache of tuned parameters
    cache: HashMap<CacheKey, KernelParams>,

    /// Whether cache has been modified
    cache_dirty: bool,
}

impl KernelTuner {
    /// Create a new kernel tuner
    pub fn new(config: TuningConfig) -> Result<Self> {
        let platform = PlatformInfo::detect()?;

        let mut tuner = Self {
            config,
            platform,
            cache: HashMap::new(),
            cache_dirty: false,
        };

        // Load cached tuning results
        tuner.load_cache()?;

        Ok(tuner)
    }

    /// Create tuner for specific backend
    pub fn for_backend(backend: Backend, config: TuningConfig) -> Result<Self> {
        let platform = match backend {
            #[cfg(feature = "cuda")]
            Backend::CUDA => PlatformInfo::cuda(0)?,
            _ => PlatformInfo::detect()?,
        };

        let mut tuner = Self {
            config,
            platform,
            cache: HashMap::new(),
            cache_dirty: false,
        };

        tuner.load_cache()?;

        Ok(tuner)
    }

    /// Get or tune parameters for matrix multiplication
    pub fn tune_matmul(&mut self, m: usize, n: usize, k: usize) -> Result<KernelParams> {
        let key = CacheKey {
            operation: Operation::MatMul,
            backend: self.platform.backend,
            device_name: self.platform.device_name.clone(),
            input_shape: vec![m, n, k],
        };

        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }

        if !self.config.enable_tuning {
            // Use heuristic defaults
            return Ok(self.default_matmul_params(m, n, k));
        }

        // Auto-tune parameters
        let params = self.auto_tune_matmul(m, n, k)?;

        self.cache.insert(key, params.clone());
        self.cache_dirty = true;

        Ok(params)
    }

    /// Auto-tune matrix multiplication parameters
    fn auto_tune_matmul(&self, m: usize, n: usize, k: usize) -> Result<KernelParams> {
        let start_time = Instant::now();
        let max_duration = Duration::from_secs_f32(self.config.max_tuning_time_secs);

        let mut best_result: Option<TuningResult> = None;

        // Search space for block sizes
        let block_sizes = vec![
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (128, 128, 8),
        ];

        // Search space for thread counts
        let thread_counts = vec![64, 128, 256, 512, 1024];

        // Search space for unroll factors
        let unroll_factors = vec![1, 2, 4, 8];

        for &block_size in &block_sizes {
            if start_time.elapsed() > max_duration {
                break;
            }

            for &threads in &thread_counts {
                if threads > self.platform.max_threads_per_block {
                    continue;
                }

                for &unroll in &unroll_factors {
                    if start_time.elapsed() > max_duration {
                        break;
                    }

                    let params = KernelParams {
                        operation: Operation::MatMul,
                        block_size,
                        threads_per_block: threads,
                        use_shared_memory: true,
                        unroll_factor: unroll,
                        vector_width: 4,
                        grid_size: self.compute_grid_size(m, n, block_size),
                        estimated_time_us: 0.0,
                    };

                    // Benchmark this configuration
                    if let Ok(result) = self.benchmark_config(&params, m, n, k) {
                        if best_result.is_none()
                            || result.mean_time < best_result.as_ref().unwrap().mean_time
                        {
                            best_result = Some(result);
                        }
                    }
                }
            }
        }

        if let Some(result) = best_result {
            let mut params = result.params;
            params.estimated_time_us = result.mean_time.as_secs_f64() * 1_000_000.0;
            Ok(params)
        } else {
            Ok(self.default_matmul_params(m, n, k))
        }
    }

    /// Benchmark a specific kernel configuration
    fn benchmark_config(
        &self,
        params: &KernelParams,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<TuningResult> {
        let mut timings = Vec::new();

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            self.execute_kernel(params, m, n, k)?;
        }

        // Benchmark iterations
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_kernel(params, m, n, k)?;
            timings.push(start.elapsed());
        }

        // Compute statistics
        let mean_time = timings.iter().sum::<Duration>() / timings.len() as u32;

        let variance = timings
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean_time.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / timings.len() as f64;

        let std_dev = variance.sqrt();

        Ok(TuningResult {
            params: params.clone(),
            mean_time,
            std_dev,
        })
    }

    /// Execute kernel with given parameters (mock implementation)
    fn execute_kernel(
        &self,
        _params: &KernelParams,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        // This would execute the actual kernel
        // For now, simulate execution time based on parameters
        std::thread::sleep(Duration::from_micros(10));
        Ok(())
    }

    /// Compute grid size for given problem and block size
    fn compute_grid_size(
        &self,
        m: usize,
        n: usize,
        block_size: (usize, usize, usize),
    ) -> (usize, usize, usize) {
        let grid_x = m.div_ceil(block_size.0);
        let grid_y = n.div_ceil(block_size.1);
        (grid_x, grid_y, 1)
    }

    /// Get default parameters for matrix multiplication
    fn default_matmul_params(&self, m: usize, n: usize, _k: usize) -> KernelParams {
        let block_size = self.platform.suggested_block_size(Operation::MatMul);

        KernelParams {
            operation: Operation::MatMul,
            block_size,
            threads_per_block: 256,
            use_shared_memory: true,
            unroll_factor: 4,
            vector_width: 4,
            grid_size: self.compute_grid_size(m, n, block_size),
            estimated_time_us: 0.0,
        }
    }

    /// Tune parameters for a generic operation
    pub fn tune_operation(
        &mut self,
        operation: Operation,
        input_shape: &[usize],
    ) -> Result<KernelParams> {
        let key = CacheKey {
            operation,
            backend: self.platform.backend,
            device_name: self.platform.device_name.clone(),
            input_shape: input_shape.to_vec(),
        };

        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }

        // Use heuristic defaults for non-matmul operations
        let block_size = self.platform.suggested_block_size(operation);

        let params = KernelParams {
            operation,
            block_size,
            threads_per_block: 256,
            use_shared_memory: matches!(
                operation,
                Operation::Attention | Operation::LayerNorm | Operation::Softmax
            ),
            unroll_factor: 4,
            vector_width: 4,
            grid_size: (1, 1, 1),
            estimated_time_us: 0.0,
        };

        self.cache.insert(key, params.clone());
        self.cache_dirty = true;

        Ok(params)
    }

    /// Load tuning cache from disk
    fn load_cache(&mut self) -> Result<()> {
        if let Some(cache_dir) = &self.config.cache_dir {
            let cache_file = cache_dir.join(format!(
                "kernel_cache_{}_{}.json",
                self.platform.backend as u8, self.platform.device_name
            ));

            if cache_file.exists() {
                let contents = std::fs::read_to_string(&cache_file).map_err(|e| {
                    TrustformersError::io_error(format!("Failed to read cache: {}", e))
                })?;

                // Deserialize from Vec and convert to HashMap
                let cache_vec: Vec<(CacheKey, KernelParams)> = serde_json::from_str(&contents)
                    .map_err(|e| {
                        TrustformersError::io_error(format!("Failed to parse cache: {}", e))
                    })?;

                self.cache = cache_vec.into_iter().collect();
            }
        }

        Ok(())
    }

    /// Save tuning cache to disk
    pub fn save_cache(&mut self) -> Result<()> {
        if !self.cache_dirty {
            return Ok(());
        }

        if let Some(cache_dir) = &self.config.cache_dir {
            std::fs::create_dir_all(cache_dir).map_err(|e| {
                TrustformersError::io_error(format!("Failed to create cache dir: {}", e))
            })?;

            let cache_file = cache_dir.join(format!(
                "kernel_cache_{}_{}.json",
                self.platform.backend as u8, self.platform.device_name
            ));

            // Convert to Vec for serialization (JSON doesn't support non-string keys)
            let cache_vec: Vec<(CacheKey, KernelParams)> =
                self.cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            let contents = serde_json::to_string_pretty(&cache_vec).map_err(|e| {
                TrustformersError::io_error(format!("Failed to serialize cache: {}", e))
            })?;

            std::fs::write(&cache_file, contents).map_err(|e| {
                TrustformersError::io_error(format!("Failed to write cache: {}", e))
            })?;

            self.cache_dirty = false;
        }

        Ok(())
    }

    /// Clear all cached tuning results
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_dirty = true;
    }

    /// Get platform information
    pub fn platform_info(&self) -> &PlatformInfo {
        &self.platform
    }

    /// Get tuning statistics
    pub fn get_statistics(&self) -> TuningStatistics {
        TuningStatistics {
            total_cached_configs: self.cache.len(),
            backends_covered: vec![self.platform.backend],
            operations_tuned: self
                .cache
                .keys()
                .map(|k| k.operation)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect(),
        }
    }
}

impl Drop for KernelTuner {
    fn drop(&mut self) {
        // Auto-save cache on drop
        let _ = self.save_cache();
    }
}

/// Statistics about tuning results
#[derive(Debug, Clone)]
pub struct TuningStatistics {
    /// Total number of cached configurations
    pub total_cached_configs: usize,

    /// Backends that have tuned configurations
    pub backends_covered: Vec<Backend>,

    /// Operations that have been tuned
    pub operations_tuned: Vec<Operation>,
}

/// Global kernel tuner instance
static mut GLOBAL_TUNER: Option<KernelTuner> = None;
static TUNER_INIT: std::sync::Once = std::sync::Once::new();

/// Get or initialize the global kernel tuner
#[allow(static_mut_refs)]
pub fn get_kernel_tuner() -> &'static mut KernelTuner {
    unsafe {
        TUNER_INIT.call_once(|| {
            GLOBAL_TUNER = Some(
                KernelTuner::new(TuningConfig::default())
                    .expect("Failed to initialize kernel tuner"),
            );
        });

        GLOBAL_TUNER.as_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() -> Result<()> {
        let platform = PlatformInfo::detect()?;

        assert!(platform.compute_units > 0);
        assert!(platform.total_memory > 0);
        assert!(!platform.device_name.is_empty());

        Ok(())
    }

    #[test]
    fn test_kernel_tuner_creation() -> Result<()> {
        let tuner = KernelTuner::new(TuningConfig::default())?;

        assert_eq!(tuner.platform.backend, Backend::CPU);

        Ok(())
    }

    #[test]
    fn test_matmul_tuning() -> Result<()> {
        let mut tuner = KernelTuner::new(TuningConfig {
            enable_tuning: false, // Use defaults for testing
            ..Default::default()
        })?;

        let params = tuner.tune_matmul(1024, 768, 512)?;

        assert_eq!(params.operation, Operation::MatMul);
        assert!(params.block_size.0 > 0);
        assert!(params.threads_per_block > 0);

        Ok(())
    }

    #[test]
    fn test_cache_persistence() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("kernel_cache_test");

        {
            let mut tuner = KernelTuner::new(TuningConfig {
                cache_dir: Some(temp_dir.clone()),
                enable_tuning: true,
                max_tuning_time_secs: 1.0, // Short tuning time for tests
                ..Default::default()
            })?;

            let _ = tuner.tune_matmul(128, 128, 128)?;
            assert!(
                tuner.cache.len() > 0,
                "Cache should be populated after tuning"
            );
            tuner.save_cache()?;
        }

        // Load cache in new instance
        {
            let tuner = KernelTuner::new(TuningConfig {
                cache_dir: Some(temp_dir.clone()),
                ..Default::default()
            })?;

            assert!(tuner.cache.len() > 0, "Cache should be loaded from disk");
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);

        Ok(())
    }

    #[test]
    fn test_operation_tuning() -> Result<()> {
        let mut tuner = KernelTuner::new(TuningConfig::default())?;

        let params = tuner.tune_operation(Operation::Softmax, &[1024, 512])?;

        assert_eq!(params.operation, Operation::Softmax);

        Ok(())
    }

    #[test]
    fn test_suggested_block_sizes() {
        let platform = PlatformInfo {
            backend: Backend::CUDA,
            device_name: "Test GPU".to_string(),
            compute_units: 80,
            total_memory: 16 * 1024 * 1024 * 1024,
            memory_bandwidth: 600.0,
            peak_tflops: 40.0,
            cache_sizes: vec![128 * 1024],
            warp_size: 32,
            max_threads_per_block: 1024,
        };

        let matmul_size = platform.suggested_block_size(Operation::MatMul);
        assert_eq!(matmul_size, (16, 16, 1));

        let softmax_size = platform.suggested_block_size(Operation::Softmax);
        assert_eq!(softmax_size, (256, 1, 1));
    }
}
