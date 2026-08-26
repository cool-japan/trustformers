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
//! ```rust,no_run
//! use trustformers_core::kernel_tuning::{KernelTuner, TuningConfig, Operation};
//!
//! // Create tuner with default configuration
//! let mut tuner = KernelTuner::new(TuningConfig::default())?;
//!
//! // Auto-tune matrix multiplication parameters for 1024x768 * 768x512
//! let params = tuner.tune_matmul(1024, 512, 768)?;
//! println!("Optimal block size: {:?}", params.block_size);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::errors::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};
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

    /// Memory bandwidth in GB/s, when the platform reports it.
    ///
    /// `None` on platforms where the value cannot be queried; it is never
    /// filled in with a guess.
    pub memory_bandwidth: Option<f32>,

    /// Peak compute performance in TFLOPS, when the platform reports it.
    ///
    /// `None` on platforms where the value cannot be queried.
    pub peak_tflops: Option<f32>,

    /// Cache sizes (L1, L2, L3) in bytes. Empty when unavailable.
    pub cache_sizes: Vec<usize>,

    /// Warp/wavefront size
    pub warp_size: usize,

    /// Maximum threads per block/workgroup
    pub max_threads_per_block: usize,
}

impl PlatformInfo {
    /// Detect current platform characteristics from the host.
    ///
    /// CPU count, CPU brand string and installed RAM come from `sysinfo`.
    /// Values the host does not expose (memory bandwidth, peak FLOPS, cache
    /// sizes) are reported as `None`/empty rather than guessed.
    pub fn detect() -> Result<Self> {
        use sysinfo::{MemoryRefreshKind, RefreshKind, System};

        let system = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
        );

        let device_name = {
            let cpu_system = System::new_with_specifics(
                RefreshKind::nothing().with_cpu(sysinfo::CpuRefreshKind::nothing()),
            );
            cpu_system
                .cpus()
                .first()
                .map(|cpu| cpu.brand().trim().to_string())
                .filter(|brand| !brand.is_empty())
                .unwrap_or_else(|| "Unknown CPU".to_string())
        };

        Ok(Self {
            backend: Backend::CPU,
            device_name,
            compute_units: num_cpus::get(),
            total_memory: usize::try_from(system.total_memory()).unwrap_or(usize::MAX),
            memory_bandwidth: None,
            peak_tflops: None,
            cache_sizes: Vec::new(),
            warp_size: 1,
            max_threads_per_block: num_cpus::get().max(1),
        })
    }

    /// Create platform info for a CUDA device.
    ///
    /// Querying real CUDA device properties requires the CUDA driver bindings,
    /// which this module does not link. Rather than reporting invented device
    /// specifications, this returns [`TrustformersError::not_implemented`].
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: usize) -> Result<Self> {
        Err(TrustformersError::not_implemented(format!(
            "CUDA device property query for device {} is not wired to the CUDA driver; \
             no device characteristics are available",
            device_id
        )))
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

/// Executes a kernel with a candidate parameter set so the tuner can time it.
///
/// The tuner never fabricates a timing: without an executor it refuses to
/// produce (or cache) a tuning result at all.
pub trait KernelExecutor: Send + Sync + std::fmt::Debug {
    /// Human-readable name, used in error messages.
    fn name(&self) -> &str;

    /// Run the kernel once with `params` for an `m x k` by `k x n` problem.
    ///
    /// Implementations must perform the real work: the tuner measures wall
    /// clock time around this call and ranks configurations by it.
    fn execute(&self, params: &KernelParams, m: usize, n: usize, k: usize) -> Result<()>;
}

/// Buffers reused across benchmark iterations so the measurement reflects the
/// kernel rather than allocation.
#[derive(Debug, Default)]
struct MatmulBuffers {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    dims: (usize, usize, usize),
}

/// A real blocked f32 GEMM used as the default CPU tuning target.
///
/// `block_size` selects the (i, j, k) tile extents and `unroll_factor` the
/// inner-loop chunk width, so different candidate parameter sets really do
/// execute different code paths and produce different timings.
#[derive(Debug, Default)]
pub struct CpuBlockedMatmulExecutor {
    buffers: Mutex<MatmulBuffers>,
}

impl CpuBlockedMatmulExecutor {
    /// Create a new CPU GEMM tuning target.
    pub fn new() -> Self {
        Self::default()
    }
}

impl KernelExecutor for CpuBlockedMatmulExecutor {
    fn name(&self) -> &str {
        "cpu_blocked_matmul_f32"
    }

    fn execute(&self, params: &KernelParams, m: usize, n: usize, k: usize) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(TrustformersError::invalid_input(
                "matmul tuning requires non-zero m, n and k".to_string(),
            ));
        }

        let mut buffers = self
            .buffers
            .lock()
            .map_err(|error| TrustformersError::lock_error(error.to_string()))?;

        if buffers.dims != (m, n, k) {
            // Deterministic, non-trivial operands so the compiler cannot fold
            // the multiplication away.
            buffers.a = (0..m * k).map(|i| ((i % 17) as f32) * 0.125 - 1.0).collect();
            buffers.b = (0..k * n).map(|i| ((i % 13) as f32) * 0.0625 - 0.5).collect();
            buffers.c = vec![0.0f32; m * n];
            buffers.dims = (m, n, k);
        }

        let block_m = params.block_size.0.max(1);
        let block_n = params.block_size.1.max(1);
        let block_k = params.block_size.2.max(1);
        let unroll = params.unroll_factor.max(1);

        let MatmulBuffers { a, b, c, .. } = &mut *buffers;
        c.fill(0.0);

        for i0 in (0..m).step_by(block_m) {
            let i_end = (i0 + block_m).min(m);
            for j0 in (0..n).step_by(block_n) {
                let j_end = (j0 + block_n).min(n);
                for p0 in (0..k).step_by(block_k) {
                    let p_end = (p0 + block_k).min(k);
                    for i in i0..i_end {
                        let row_c = i * n;
                        let row_a = i * k;
                        for p in p0..p_end {
                            let a_value = a[row_a + p];
                            let row_b = p * n;
                            let mut j = j0;
                            while j + unroll <= j_end {
                                for offset in 0..unroll {
                                    c[row_c + j + offset] += a_value * b[row_b + j + offset];
                                }
                                j += unroll;
                            }
                            while j < j_end {
                                c[row_c + j] += a_value * b[row_b + j];
                                j += 1;
                            }
                        }
                    }
                }
            }
        }

        // Keep the result observable so the work cannot be optimised out.
        std::hint::black_box(&c[0]);
        Ok(())
    }
}

/// Tuning result for a specific configuration
#[derive(Debug, Clone)]
struct TuningResult {
    params: KernelParams,
    mean_time: Duration,
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
#[derive(Debug)]
pub struct KernelTuner {
    /// Tuning configuration
    config: TuningConfig,

    /// Platform information
    platform: PlatformInfo,

    /// Cache of tuned parameters
    cache: HashMap<CacheKey, KernelParams>,

    /// Whether cache has been modified
    cache_dirty: bool,

    /// The kernel actually timed while searching the parameter space.
    executor: Option<Box<dyn KernelExecutor>>,
}

impl KernelTuner {
    /// Create a new kernel tuner.
    ///
    /// A CPU platform gets the in-tree blocked GEMM
    /// ([`CpuBlockedMatmulExecutor`]) as its tuning target. Other backends
    /// start without an executor; call [`Self::with_executor`] before enabling
    /// tuning, otherwise `tune_matmul` reports that no kernel can be measured.
    pub fn new(config: TuningConfig) -> Result<Self> {
        let platform = PlatformInfo::detect()?;
        Self::with_platform(config, platform)
    }

    fn with_platform(config: TuningConfig, platform: PlatformInfo) -> Result<Self> {
        let executor: Option<Box<dyn KernelExecutor>> = match platform.backend {
            Backend::CPU => Some(Box::new(CpuBlockedMatmulExecutor::new())),
            _ => None,
        };

        let mut tuner = Self {
            config,
            platform,
            cache: HashMap::new(),
            cache_dirty: false,
            executor,
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

        Self::with_platform(config, platform)
    }

    /// Install the kernel the tuner should benchmark.
    pub fn with_executor(mut self, executor: Box<dyn KernelExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Install the kernel the tuner should benchmark, in place.
    pub fn set_executor(&mut self, executor: Box<dyn KernelExecutor>) {
        self.executor = Some(executor);
    }

    /// Name of the kernel currently wired up for benchmarking, if any.
    pub fn executor_name(&self) -> Option<&str> {
        self.executor.as_ref().map(|executor| executor.name())
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

    /// Auto-tune matrix multiplication parameters by timing the installed
    /// kernel executor over the candidate parameter space.
    ///
    /// Only parameters the executor can act on are searched: the tile extents
    /// and the inner-loop unroll factor. `threads_per_block` is taken from the
    /// platform, because ranking configurations by a parameter the kernel
    /// ignores would be ranking scheduler jitter.
    fn auto_tune_matmul(&self, m: usize, n: usize, k: usize) -> Result<KernelParams> {
        let executor = self.executor.as_ref().ok_or_else(|| {
            TrustformersError::not_implemented(format!(
                "no kernel executor is wired up for backend {:?}; \
                 install one with KernelTuner::with_executor before enabling tuning, \
                 or set TuningConfig::enable_tuning = false to use heuristic defaults",
                self.platform.backend
            ))
        })?;

        let start_time = Instant::now();
        let max_duration = Duration::from_secs_f32(self.config.max_tuning_time_secs);

        let mut best_result: Option<TuningResult> = None;
        let mut measured_configurations = 0usize;

        // Search space for block sizes
        let block_sizes = [
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (128, 128, 8),
        ];

        // Search space for unroll factors
        let unroll_factors = [1usize, 2, 4, 8];

        let threads_per_block = self
            .platform
            .max_threads_per_block
            .clamp(1, self.platform.max_threads_per_block.max(1));

        'search: for &block_size in &block_sizes {
            for &unroll in &unroll_factors {
                if start_time.elapsed() > max_duration && best_result.is_some() {
                    break 'search;
                }

                let params = KernelParams {
                    operation: Operation::MatMul,
                    block_size,
                    threads_per_block,
                    use_shared_memory: true,
                    unroll_factor: unroll,
                    vector_width: 4,
                    grid_size: self.compute_grid_size(m, n, block_size),
                    estimated_time_us: 0.0,
                };

                // Benchmark this configuration against the real kernel.
                let result = self.benchmark_config(executor.as_ref(), &params, m, n, k)?;
                measured_configurations += 1;
                let is_better = match &best_result {
                    None => true,
                    Some(best) => result.mean_time < best.mean_time,
                };
                if is_better {
                    best_result = Some(result);
                }
            }
        }

        let result = best_result.ok_or_else(|| {
            TrustformersError::runtime_error(
                "kernel tuning produced no measurement; refusing to cache a tuning result"
                    .to_string(),
            )
        })?;

        debug_assert!(measured_configurations > 0);
        let mut params = result.params;
        params.estimated_time_us = result.mean_time.as_secs_f64() * 1_000_000.0;
        Ok(params)
    }

    /// Benchmark a specific kernel configuration.
    fn benchmark_config(
        &self,
        executor: &dyn KernelExecutor,
        params: &KernelParams,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<TuningResult> {
        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            executor.execute(params, m, n, k)?;
        }

        // Benchmark iterations, timed around the real kernel invocation.
        let iterations = self.config.benchmark_iterations.max(1);
        let mut timings = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            executor.execute(params, m, n, k)?;
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

    /// Measure one parameter set against the installed executor.
    ///
    /// Returns the mean wall-clock time and its standard deviation over
    /// [`TuningConfig::benchmark_iterations`] runs.
    pub fn measure(
        &self,
        params: &KernelParams,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(Duration, f64)> {
        let executor = self.executor.as_ref().ok_or_else(|| {
            TrustformersError::not_implemented(
                "no kernel executor is wired up; nothing can be measured".to_string(),
            )
        })?;
        let result = self.benchmark_config(executor.as_ref(), params, m, n, k)?;
        Ok((result.mean_time, result.std_dev))
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

    /// Heuristic parameters for a generic operation.
    ///
    /// No benchmark is run for non-matmul operations (this crate has no
    /// executor for them), so the result is *not* written into the tuning
    /// cache: the cache only ever holds measured configurations.
    /// `estimated_time_us` stays `0.0` because nothing was measured.
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

        Ok(KernelParams {
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
        })
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

/// Global kernel tuner instance.
///
/// A `OnceLock<Mutex<_>>` rather than a `static mut`: the previous accessor
/// handed out aliasing `&'static mut` references from safe code, which is
/// undefined behaviour and a data race on the tuning cache.
static GLOBAL_TUNER: OnceLock<Mutex<KernelTuner>> = OnceLock::new();

/// Get the global kernel tuner, locking it for exclusive use.
///
/// Returns an error if the tuner could not be constructed (for example because
/// the platform could not be detected) or if the lock is poisoned.
pub fn get_kernel_tuner() -> Result<MutexGuard<'static, KernelTuner>> {
    // `OnceLock::get_or_init` cannot fail, so construct fallibly first and
    // store only a successfully built tuner.
    if GLOBAL_TUNER.get().is_none() {
        let tuner = KernelTuner::new(TuningConfig::default())?;
        // A concurrent initialiser may win the race; that is fine, the loser's
        // tuner is simply dropped.
        let _ = GLOBAL_TUNER.set(Mutex::new(tuner));
    }

    let tuner = GLOBAL_TUNER.get().ok_or_else(|| {
        TrustformersError::runtime_error("global kernel tuner is not initialised".to_string())
    })?;

    tuner.lock().map_err(|error| {
        TrustformersError::lock_error(format!("global kernel tuner mutex poisoned: {}", error))
    })
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

    fn test_cache_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "trustformers_kernel_cache_{}_{}",
            name,
            std::process::id()
        ))
    }

    #[test]
    fn test_cache_persistence() -> Result<()> {
        let temp_dir = test_cache_dir("persistence");
        let _ = std::fs::remove_dir_all(&temp_dir);

        {
            let mut tuner = KernelTuner::new(TuningConfig {
                cache_dir: Some(temp_dir.clone()),
                enable_tuning: true,
                max_tuning_time_secs: 1.0, // Short tuning time for tests
                warmup_iterations: 1,
                benchmark_iterations: 2,
                ..Default::default()
            })?;

            let _ = tuner.tune_matmul(48, 48, 48)?;
            assert!(
                !tuner.cache.is_empty(),
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

            assert!(!tuner.cache.is_empty(), "Cache should be loaded from disk");
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);

        Ok(())
    }

    /// Regression test: `execute_kernel` used to `thread::sleep(10us)` and
    /// ignore every parameter, so all configurations timed identically and the
    /// "optimal" configuration was scheduler jitter. Timing must now reflect
    /// the actual amount of work.
    #[test]
    fn test_tuning_measures_real_kernel_work() -> Result<()> {
        let tuner = KernelTuner::new(TuningConfig {
            cache_dir: None,
            enable_tuning: true,
            warmup_iterations: 1,
            benchmark_iterations: 3,
            ..Default::default()
        })?;

        assert_eq!(tuner.executor_name(), Some("cpu_blocked_matmul_f32"));

        let params = KernelParams {
            operation: Operation::MatMul,
            block_size: (32, 32, 32),
            ..Default::default()
        };

        let (small_time, _) = tuner.measure(&params, 16, 16, 16)?;
        let (large_time, _) = tuner.measure(&params, 96, 96, 96)?;

        // 96^3 is 216x the work of 16^3. A fixed sleep would make these equal.
        assert!(
            large_time > small_time * 4,
            "a 216x larger GEMM must take measurably longer: {:?} vs {:?}",
            large_time,
            small_time
        );
        assert!(
            small_time < Duration::from_millis(50),
            "a 16x16x16 GEMM must not take the old fixed 10us sleep path"
        );

        Ok(())
    }

    /// Regression test: without a kernel to time, the tuner must refuse rather
    /// than cache a configuration ranked on nothing.
    #[test]
    fn test_tuning_without_executor_is_refused() -> Result<()> {
        let mut tuner = KernelTuner::new(TuningConfig {
            cache_dir: None,
            enable_tuning: true,
            ..Default::default()
        })?;
        tuner.executor = None;

        let error = tuner.tune_matmul(32, 32, 32).expect_err("must not tune without an executor");
        assert!(
            error.to_string().contains("no kernel executor"),
            "unexpected error: {}",
            error
        );
        assert!(
            tuner.cache.is_empty(),
            "nothing may be cached when nothing was measured"
        );

        Ok(())
    }

    /// Regression test: `tune_operation` used to cache an unmeasured heuristic
    /// as if it were a tuning result.
    #[test]
    fn test_operation_tuning() -> Result<()> {
        let mut tuner = KernelTuner::new(TuningConfig {
            cache_dir: None,
            ..Default::default()
        })?;

        let params = tuner.tune_operation(Operation::Softmax, &[1024, 512])?;

        assert_eq!(params.operation, Operation::Softmax);
        assert_eq!(
            params.estimated_time_us, 0.0,
            "no measurement was taken, so no time may be reported"
        );
        assert!(
            tuner.cache.is_empty(),
            "heuristic parameters must not enter the measured-configuration cache"
        );

        Ok(())
    }

    /// Regression test: `get_kernel_tuner` used to hand out `&'static mut` from
    /// a `static mut`. It must now hand out a guarded, shareable handle.
    #[test]
    fn test_global_tuner_is_mutex_guarded() -> Result<()> {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let mut tuner = get_kernel_tuner().expect("global tuner");
                    let params = tuner.tune_operation(Operation::LayerNorm, &[128])?;
                    assert_eq!(params.operation, Operation::LayerNorm);
                    Ok::<(), TrustformersError>(())
                })
            })
            .collect();

        for handle in handles {
            handle.join().map_err(|_| {
                TrustformersError::runtime_error("tuner thread panicked".to_string())
            })??;
        }

        Ok(())
    }

    #[test]
    fn test_platform_detection_reports_no_invented_specs() -> Result<()> {
        let platform = PlatformInfo::detect()?;
        assert!(
            platform.memory_bandwidth.is_none(),
            "memory bandwidth is not queryable and must not be invented"
        );
        assert!(platform.peak_tflops.is_none());
        assert!(platform.total_memory > 0, "total memory comes from sysinfo");
        Ok(())
    }

    #[test]
    fn test_suggested_block_sizes() {
        let platform = PlatformInfo {
            backend: Backend::CUDA,
            device_name: "Test GPU".to_string(),
            compute_units: 80,
            total_memory: 16 * 1024 * 1024 * 1024,
            memory_bandwidth: None,
            peak_tflops: None,
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
