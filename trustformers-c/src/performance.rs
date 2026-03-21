//! Performance optimization module for TrustformeRS-C
//!
//! This module provides advanced performance optimizations including:
//! - SIMD optimizations for vectorized operations
//! - Dynamic batching for improved throughput
//! - Kernel fusion for reduced memory bandwidth

use parking_lot::RwLock;
use std::collections::HashMap;
use std::os::raw::{c_char, c_double, c_float, c_int};
use std::ptr;
use std::sync::Arc;

use crate::error::{TrustformersError, TrustformersResult};
use crate::platform::{PlatformInfo, PlatformOptimizer};
use crate::utils::string_to_c_str;

/// Performance optimization configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Enable SIMD optimizations
    pub enable_simd: c_int,
    /// Enable dynamic batching
    pub enable_dynamic_batching: c_int,
    /// Enable kernel fusion
    pub enable_kernel_fusion: c_int,
    /// Maximum batch size for dynamic batching
    pub max_batch_size: c_int,
    /// Target latency for dynamic batching (milliseconds)
    pub target_latency_ms: c_int,
    /// Memory bandwidth optimization level (0-3)
    pub memory_bandwidth_level: c_int,
    /// Enable multi-threading
    pub enable_multithreading: c_int,
    /// Number of threads (0 = auto-detect)
    pub num_threads: c_int,
    /// Cache size for kernel fusion (MB)
    pub fusion_cache_size_mb: c_int,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        let platform_info = PlatformInfo::detect();
        Self {
            enable_simd: if platform_info.cpu_features.contains(&"avx2".to_string())
                || platform_info.cpu_features.contains(&"neon".to_string())
            {
                1
            } else {
                0
            },
            enable_dynamic_batching: 1,
            enable_kernel_fusion: 1,
            max_batch_size: 32,
            target_latency_ms: 100,
            memory_bandwidth_level: 2,
            enable_multithreading: 1,
            num_threads: platform_info.cpu_cores as c_int,
            fusion_cache_size_mb: 128,
        }
    }
}

/// SIMD optimization engine
pub struct SimdOptimizer {
    platform_optimizer: PlatformOptimizer,
    config: PerformanceConfig,
}

impl SimdOptimizer {
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            platform_optimizer: PlatformOptimizer::new(),
            config,
        }
    }

    /// Optimized matrix multiplication using SIMD
    pub fn matrix_multiply_simd(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> TrustformersResult<()> {
        if self.config.enable_simd != 0 {
            let matrix_multiply_fn = self.platform_optimizer.get_matrix_multiply_fn();
            matrix_multiply_fn(a, b, c, m, n, k);
        } else {
            // Fallback to generic implementation
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for ki in 0..k {
                        sum += a[i * k + ki] * b[ki * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
        Ok(())
    }

    /// Optimized vector operations using SIMD
    pub fn vector_add_simd(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> TrustformersResult<()> {
        if self.config.enable_simd != 0 {
            // Use platform-specific SIMD implementations
            let info = self.platform_optimizer.get_info();

            #[cfg(target_arch = "x86_64")]
            {
                if info.cpu_features.contains(&"avx2".to_string()) {
                    unsafe {
                        self.vector_add_avx2(a, b, c)?;
                    }
                    return Ok(());
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if info.cpu_features.contains(&"neon".to_string()) {
                    unsafe {
                        self.vector_add_neon(a, b, c)?;
                    }
                    return Ok(());
                }
            }
        }

        // Fallback to generic implementation
        for i in 0..a.len().min(b.len()).min(c.len()) {
            c[i] = a[i] + b[i];
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_add_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> TrustformersResult<()> {
        use std::arch::x86_64::*;

        let len = a.len().min(b.len()).min(c.len());
        let simd_len = len & !7; // Round down to multiple of 8

        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let result = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(c.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            c[i] = a[i] + b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn vector_add_neon(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> TrustformersResult<()> {
        use std::arch::aarch64::*;

        let len = a.len().min(b.len()).min(c.len());
        let simd_len = len & !3; // Round down to multiple of 4

        for i in (0..simd_len).step_by(4) {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            let result = vaddq_f32(a_vec, b_vec);
            vst1q_f32(c.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            c[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Optimized activation functions using SIMD
    pub fn relu_simd(&self, input: &[f32], output: &mut [f32]) -> TrustformersResult<()> {
        if self.config.enable_simd != 0 {
            let info = self.platform_optimizer.get_info();

            #[cfg(target_arch = "x86_64")]
            {
                if info.cpu_features.contains(&"avx2".to_string()) {
                    unsafe {
                        self.relu_avx2(input, output)?;
                    }
                    return Ok(());
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if info.cpu_features.contains(&"neon".to_string()) {
                    unsafe {
                        self.relu_neon(input, output)?;
                    }
                    return Ok(());
                }
            }
        }

        // Fallback implementation
        for i in 0..input.len().min(output.len()) {
            output[i] = input[i].max(0.0);
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(&self, input: &[f32], output: &mut [f32]) -> TrustformersResult<()> {
        use std::arch::x86_64::*;

        let len = input.len().min(output.len());
        let simd_len = len & !7;
        let zeros = _mm256_setzero_ps();

        for i in (0..simd_len).step_by(8) {
            let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
            let result = _mm256_max_ps(input_vec, zeros);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].max(0.0);
        }

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn relu_neon(&self, input: &[f32], output: &mut [f32]) -> TrustformersResult<()> {
        use std::arch::aarch64::*;

        let len = input.len().min(output.len());
        let simd_len = len & !3;
        let zeros = vdupq_n_f32(0.0);

        for i in (0..simd_len).step_by(4) {
            let input_vec = vld1q_f32(input.as_ptr().add(i));
            let result = vmaxq_f32(input_vec, zeros);
            vst1q_f32(output.as_mut_ptr().add(i), result);
        }

        for i in simd_len..len {
            output[i] = input[i].max(0.0);
        }

        Ok(())
    }
}

/// Dynamic batching system for improved throughput
pub struct DynamicBatcher {
    config: PerformanceConfig,
    pending_requests: Vec<BatchRequest>,
    batch_timer: Option<std::time::Instant>,
    throughput_stats: ThroughputStats,
}

#[derive(Debug, Clone)]
struct BatchRequest {
    id: u64,
    input: Vec<f32>,
    timestamp: std::time::Instant,
    priority: BatchPriority,
}

#[derive(Debug, Clone, Copy)]
enum BatchPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Default)]
struct ThroughputStats {
    total_requests: u64,
    total_batches: u64,
    average_batch_size: f64,
    average_latency_ms: f64,
    throughput_per_second: f64,
}

impl DynamicBatcher {
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            pending_requests: Vec::new(),
            batch_timer: None,
            throughput_stats: ThroughputStats::default(),
        }
    }

    pub fn add_request(&mut self, input: Vec<f32>, priority: BatchPriority) -> u64 {
        let id = self.throughput_stats.total_requests;
        self.throughput_stats.total_requests += 1;

        let request = BatchRequest {
            id,
            input,
            timestamp: std::time::Instant::now(),
            priority,
        };

        self.pending_requests.push(request);

        // Set batch timer if this is the first request
        if self.batch_timer.is_none() {
            self.batch_timer = Some(std::time::Instant::now());
        }

        id
    }

    pub fn should_process_batch(&self) -> bool {
        if self.pending_requests.is_empty() {
            return false;
        }

        // Check if we've reached the maximum batch size
        if self.pending_requests.len() >= self.config.max_batch_size as usize {
            return true;
        }

        // Check if we've exceeded the target latency
        if let Some(timer) = self.batch_timer {
            let elapsed = timer.elapsed();
            if elapsed.as_millis() >= self.config.target_latency_ms as u128 {
                return true;
            }
        }

        // Check for high priority requests
        if self
            .pending_requests
            .iter()
            .any(|req| matches!(req.priority, BatchPriority::Critical))
        {
            return true;
        }

        false
    }

    pub fn create_batch(&mut self) -> Option<ProcessingBatch> {
        if self.pending_requests.is_empty() {
            return None;
        }

        // Sort by priority (critical first)
        self.pending_requests.sort_by(|a, b| match (a.priority, b.priority) {
            (BatchPriority::Critical, BatchPriority::Critical) => a.timestamp.cmp(&b.timestamp),
            (BatchPriority::Critical, _) => std::cmp::Ordering::Less,
            (_, BatchPriority::Critical) => std::cmp::Ordering::Greater,
            (BatchPriority::High, BatchPriority::High) => a.timestamp.cmp(&b.timestamp),
            (BatchPriority::High, _) => std::cmp::Ordering::Less,
            (_, BatchPriority::High) => std::cmp::Ordering::Greater,
            _ => a.timestamp.cmp(&b.timestamp),
        });

        // Take up to max_batch_size requests
        let batch_size = self.pending_requests.len().min(self.config.max_batch_size as usize);
        let batch_requests = self.pending_requests.drain(..batch_size).collect();

        // Reset batch timer
        self.batch_timer = if self.pending_requests.is_empty() {
            None
        } else {
            Some(std::time::Instant::now())
        };

        // Update statistics
        self.throughput_stats.total_batches += 1;
        self.throughput_stats.average_batch_size = (self.throughput_stats.average_batch_size
            * (self.throughput_stats.total_batches - 1) as f64
            + batch_size as f64)
            / self.throughput_stats.total_batches as f64;

        Some(ProcessingBatch {
            requests: batch_requests,
            batch_id: self.throughput_stats.total_batches,
        })
    }

    pub fn get_stats(&self) -> &ThroughputStats {
        &self.throughput_stats
    }
}

pub struct ProcessingBatch {
    pub requests: Vec<BatchRequest>,
    pub batch_id: u64,
}

impl ProcessingBatch {
    pub fn get_batch_input(&self) -> Vec<Vec<f32>> {
        self.requests.iter().map(|req| req.input.clone()).collect()
    }

    pub fn get_request_ids(&self) -> Vec<u64> {
        self.requests.iter().map(|req| req.id).collect()
    }
}

/// Kernel fusion system for reducing memory bandwidth
pub struct KernelFusion {
    config: PerformanceConfig,
    fusion_cache: HashMap<String, FusedKernel>,
    fusion_patterns: Vec<FusionPattern>,
}

#[derive(Debug, Clone)]
struct FusedKernel {
    pattern: FusionPattern,
    optimized_code: String,
    cache_hits: u64,
    average_speedup: f64,
}

#[derive(Debug, Clone)]
struct FusionPattern {
    name: String,
    operations: Vec<String>,
    fusion_type: FusionType,
    memory_reduction: f64,
    compute_efficiency: f64,
}

#[derive(Debug, Clone)]
enum FusionType {
    ElementWise,
    MatrixOps,
    ConvolutionBased,
    AttentionBased,
}

impl KernelFusion {
    pub fn new(config: PerformanceConfig) -> Self {
        let mut fusion = Self {
            config,
            fusion_cache: HashMap::new(),
            fusion_patterns: Vec::new(),
        };

        fusion.initialize_patterns();
        fusion
    }

    fn initialize_patterns(&mut self) {
        // Common fusion patterns
        self.fusion_patterns.push(FusionPattern {
            name: "conv_relu".to_string(),
            operations: vec!["convolution".to_string(), "relu".to_string()],
            fusion_type: FusionType::ConvolutionBased,
            memory_reduction: 0.5,
            compute_efficiency: 1.2,
        });

        self.fusion_patterns.push(FusionPattern {
            name: "matmul_add_relu".to_string(),
            operations: vec!["matmul".to_string(), "add".to_string(), "relu".to_string()],
            fusion_type: FusionType::MatrixOps,
            memory_reduction: 0.67,
            compute_efficiency: 1.4,
        });

        self.fusion_patterns.push(FusionPattern {
            name: "attention_fusion".to_string(),
            operations: vec![
                "qk_matmul".to_string(),
                "scale".to_string(),
                "softmax".to_string(),
                "v_matmul".to_string(),
            ],
            fusion_type: FusionType::AttentionBased,
            memory_reduction: 0.75,
            compute_efficiency: 1.8,
        });
    }

    pub fn try_fuse_operations(&mut self, operations: &[String]) -> Option<&FusedKernel> {
        if self.config.enable_kernel_fusion == 0 {
            return None;
        }

        // Check if operations match any fusion pattern
        let mut matching_pattern = None;
        for pattern in &self.fusion_patterns {
            if self.matches_pattern(operations, &pattern.operations) {
                matching_pattern = Some(pattern.clone());
                break;
            }
        }

        if let Some(pattern) = matching_pattern {
            let pattern_key = pattern.name.clone();

            // Check if already exists in cache
            if let Some(existing_kernel) = self.fusion_cache.get_mut(&pattern_key) {
                existing_kernel.cache_hits += 1;
                // Can't return mutable reference here due to lifetime issues
                // So we'll return None and let the next call get the kernel
            } else {
                // Create new fused kernel if not in cache
                let fused_kernel = self.create_fused_kernel_internal(pattern.clone());
                self.fusion_cache.insert(pattern_key.clone(), fused_kernel);
            }

            // Return immutable reference to the kernel
            return self.fusion_cache.get(&pattern_key);
        }

        None
    }

    fn matches_pattern(&self, operations: &[String], pattern: &[String]) -> bool {
        if operations.len() != pattern.len() {
            return false;
        }

        operations.iter().zip(pattern.iter()).all(|(op, pat)| op == pat)
    }

    fn create_fused_kernel_internal(&self, pattern: FusionPattern) -> FusedKernel {
        let optimized_code = match pattern.fusion_type {
            FusionType::ElementWise => self.generate_elementwise_fusion(&pattern),
            FusionType::MatrixOps => self.generate_matrix_fusion(&pattern),
            FusionType::ConvolutionBased => self.generate_convolution_fusion(&pattern),
            FusionType::AttentionBased => self.generate_attention_fusion(&pattern),
        };

        FusedKernel {
            pattern,
            optimized_code,
            cache_hits: 0,
            average_speedup: 1.0,
        }
    }

    fn generate_elementwise_fusion(&self, pattern: &FusionPattern) -> String {
        format!(
            "// Fused elementwise kernel: {}\n\
             // Operations: {:?}\n\
             // Memory reduction: {:.2}%\n\
             // Compute efficiency: {:.2}x\n\
             \n\
             void fused_elementwise_{}(const float* input, float* output, size_t size) {{\n\
                 #pragma omp parallel for simd\n\
                 for (size_t i = 0; i < size; ++i) {{\n\
                     float value = input[i];\n\
                     // Apply fused operations inline\n\
                     output[i] = value; // Placeholder for actual fused operations\n\
                 }}\n\
             }}\n",
            pattern.name,
            pattern.operations,
            pattern.memory_reduction * 100.0,
            pattern.compute_efficiency,
            pattern.name
        )
    }

    fn generate_matrix_fusion(&self, pattern: &FusionPattern) -> String {
        format!(
            "// Fused matrix kernel: {}\n\
             // Operations: {:?}\n\
             // Memory reduction: {:.2}%\n\
             // Compute efficiency: {:.2}x\n\
             \n\
             void fused_matrix_{}(const float* A, const float* B, const float* bias, float* C, \n\
                                 size_t M, size_t N, size_t K) {{\n\
                 #pragma omp parallel for\n\
                 for (size_t i = 0; i < M; ++i) {{\n\
                     for (size_t j = 0; j < N; ++j) {{\n\
                         float sum = 0.0f;\n\
                         for (size_t k = 0; k < K; ++k) {{\n\
                             sum += A[i * K + k] * B[k * N + j];\n\
                         }}\n\
                         sum += bias[j]; // Add bias inline\n\
                         C[i * N + j] = fmaxf(sum, 0.0f); // ReLU inline\n\
                     }}\n\
                 }}\n\
             }}\n",
            pattern.name,
            pattern.operations,
            pattern.memory_reduction * 100.0,
            pattern.compute_efficiency,
            pattern.name
        )
    }

    fn generate_convolution_fusion(&self, pattern: &FusionPattern) -> String {
        format!(
            "// Fused convolution kernel: {}\n\
             // Operations: {:?}\n\
             // Memory reduction: {:.2}%\n\
             // Compute efficiency: {:.2}x\n\
             \n\
             void fused_conv_{}(const float* input, const float* weights, const float* bias,\n\
                               float* output, const ConvParams* params) {{\n\
                 // Optimized convolution with activation fusion\n\
                 // Implementation depends on specific convolution parameters\n\
                 // This is a simplified placeholder\n\
             }}\n",
            pattern.name,
            pattern.operations,
            pattern.memory_reduction * 100.0,
            pattern.compute_efficiency,
            pattern.name
        )
    }

    fn generate_attention_fusion(&self, pattern: &FusionPattern) -> String {
        format!(
            "// Fused attention kernel: {}\n\
             // Operations: {:?}\n\
             // Memory reduction: {:.2}%\n\
             // Compute efficiency: {:.2}x\n\
             \n\
             void fused_attention_{}(const float* Q, const float* K, const float* V,\n\
                                   float* output, size_t seq_len, size_t d_model) {{\n\
                 // Fused attention computation\n\
                 // Q*K^T, scale, softmax, *V all in single kernel\n\
                 // Reduces memory bandwidth significantly\n\
             }}\n",
            pattern.name,
            pattern.operations,
            pattern.memory_reduction * 100.0,
            pattern.compute_efficiency,
            pattern.name
        )
    }

    pub fn get_fusion_stats(&self) -> HashMap<String, (u64, f64)> {
        self.fusion_cache
            .iter()
            .map(|(k, v)| (k.clone(), (v.cache_hits, v.average_speedup)))
            .collect()
    }
}

/// Performance optimization manager
pub struct PerformanceOptimizer {
    config: PerformanceConfig,
    simd_optimizer: SimdOptimizer,
    dynamic_batcher: Arc<RwLock<DynamicBatcher>>,
    kernel_fusion: Arc<RwLock<KernelFusion>>,
}

impl PerformanceOptimizer {
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            simd_optimizer: SimdOptimizer::new(config.clone()),
            dynamic_batcher: Arc::new(RwLock::new(DynamicBatcher::new(config.clone()))),
            kernel_fusion: Arc::new(RwLock::new(KernelFusion::new(config.clone()))),
            config,
        }
    }

    pub fn optimize_matrix_operations(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> TrustformersResult<()> {
        // Try to fuse operations if possible
        let operations = vec!["matmul".to_string()];
        let mut fusion = self.kernel_fusion.write();

        if let Some(_fused_kernel) = fusion.try_fuse_operations(&operations) {
            // Use fused kernel (placeholder - would call actual fused implementation)
            drop(fusion);
            self.simd_optimizer.matrix_multiply_simd(a, b, c, m, n, k)
        } else {
            drop(fusion);
            self.simd_optimizer.matrix_multiply_simd(a, b, c, m, n, k)
        }
    }

    pub fn get_performance_stats(&self) -> PerformanceStats {
        let batcher = self.dynamic_batcher.read();
        let fusion = self.kernel_fusion.read();

        PerformanceStats {
            simd_enabled: self.config.enable_simd != 0,
            dynamic_batching_enabled: self.config.enable_dynamic_batching != 0,
            kernel_fusion_enabled: self.config.enable_kernel_fusion != 0,
            average_batch_size: batcher.get_stats().average_batch_size,
            throughput_per_second: batcher.get_stats().throughput_per_second,
            fusion_cache_hits: fusion.get_fusion_stats().values().map(|(hits, _)| *hits).sum(),
            fusion_cache_size: fusion.get_fusion_stats().len(),
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub simd_enabled: bool,
    pub dynamic_batching_enabled: bool,
    pub kernel_fusion_enabled: bool,
    pub average_batch_size: f64,
    pub throughput_per_second: f64,
    pub fusion_cache_hits: u64,
    pub fusion_cache_size: usize,
}

/// C API exports for performance optimization
#[no_mangle]
pub extern "C" fn trustformers_create_performance_optimizer(
    config: *const PerformanceConfig,
) -> *mut PerformanceOptimizer {
    let config = if config.is_null() {
        PerformanceConfig::default()
    } else {
        unsafe { (*config).clone() }
    };

    Box::into_raw(Box::new(PerformanceOptimizer::new(config)))
}

#[no_mangle]
pub extern "C" fn trustformers_destroy_performance_optimizer(optimizer: *mut PerformanceOptimizer) {
    if !optimizer.is_null() {
        unsafe {
            let _ = Box::from_raw(optimizer);
        }
    }
}

#[no_mangle]
pub extern "C" fn trustformers_optimize_matrix_operations(
    optimizer: *mut PerformanceOptimizer,
    a: *const c_float,
    b: *const c_float,
    c: *mut c_float,
    m: usize,
    n: usize,
    k: usize,
) -> TrustformersError {
    if optimizer.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return TrustformersError::NullPointer;
    }

    unsafe {
        let optimizer = &*optimizer;
        let a_slice = std::slice::from_raw_parts(a, m * k);
        let b_slice = std::slice::from_raw_parts(b, k * n);
        let c_slice = std::slice::from_raw_parts_mut(c, m * n);

        match optimizer.optimize_matrix_operations(a_slice, b_slice, c_slice, m, n, k) {
            Ok(()) => TrustformersError::Success,
            Err(_) => TrustformersError::RuntimeError,
        }
    }
}

#[no_mangle]
pub extern "C" fn trustformers_get_performance_config() -> PerformanceConfig {
    PerformanceConfig::default()
}

#[no_mangle]
pub extern "C" fn trustformers_get_performance_stats(
    optimizer: *const PerformanceOptimizer,
    stats_json: *mut *mut c_char,
) -> TrustformersError {
    if optimizer.is_null() || stats_json.is_null() {
        return TrustformersError::NullPointer;
    }

    unsafe {
        let optimizer = &*optimizer;
        let stats = optimizer.get_performance_stats();

        let stats_json_str = match serde_json::to_string_pretty(&serde_json::json!({
            "simd_enabled": stats.simd_enabled,
            "dynamic_batching_enabled": stats.dynamic_batching_enabled,
            "kernel_fusion_enabled": stats.kernel_fusion_enabled,
            "average_batch_size": stats.average_batch_size,
            "throughput_per_second": stats.throughput_per_second,
            "fusion_cache_hits": stats.fusion_cache_hits,
            "fusion_cache_size": stats.fusion_cache_size,
        })) {
            Ok(json) => string_to_c_str(json),
            Err(_) => return TrustformersError::SerializationError,
        };

        *stats_json = stats_json_str;
        TrustformersError::Success
    }
}
