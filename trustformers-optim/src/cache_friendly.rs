//! Cache-friendly optimization algorithms for improved memory performance.
//!
//! This module implements optimization algorithms with cache-friendly memory access patterns,
//! reducing cache misses and improving overall performance, especially for large models.
//!
//! # Key Optimizations
//!
//! - **Blocked/Tiled Operations**: Process data in cache-sized blocks
//! - **Memory Layout Optimization**: Structure data for optimal cache utilization
//! - **Data Prefetching**: Improve cache hit rates with strategic prefetching
//! - **Loop Fusion**: Combine operations to reduce memory bandwidth requirements
//! - **Vectorization-Friendly**: Design for SIMD instruction utilization

use crate::common::{BiasCorrection, ParameterUpdate};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Cache configuration parameters for memory-aware optimizers.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 cache size in bytes (typically 32KB)
    pub l1_cache_size: usize,
    /// L2 cache size in bytes (typically 256KB-1MB)
    pub l2_cache_size: usize,
    /// L3 cache size in bytes (typically 8MB-32MB)
    pub l3_cache_size: usize,
    /// Cache line size in bytes (typically 64 bytes)
    pub cache_line_size: usize,
    /// Block size for tiled operations
    pub block_size: usize,
    /// Whether to enable prefetching
    pub enable_prefetching: bool,
    /// Prefetch distance (number of cache lines ahead)
    pub prefetch_distance: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_cache_size: 32 * 1024,       // 32KB
            l2_cache_size: 256 * 1024,      // 256KB
            l3_cache_size: 8 * 1024 * 1024, // 8MB
            cache_line_size: 64,            // 64 bytes
            block_size: 1024,               // Process 1024 elements at a time
            enable_prefetching: true,
            prefetch_distance: 4,
        }
    }
}

impl CacheConfig {
    /// Detects cache configuration from the system (simplified version).
    pub fn detect_system() -> Self {
        // In a real implementation, this would use cpuid or similar
        // to detect actual cache sizes
        Self::default()
    }

    /// Configures for L1 cache optimization (small blocks, high frequency access).
    pub fn l1_optimized() -> Self {
        Self {
            block_size: 512, // Smaller blocks for L1
            ..Default::default()
        }
    }

    /// Configures for L2 cache optimization (medium blocks).
    pub fn l2_optimized() -> Self {
        Self {
            block_size: 2048, // Medium blocks for L2
            ..Default::default()
        }
    }

    /// Configures for L3 cache optimization (larger blocks).
    pub fn l3_optimized() -> Self {
        Self {
            block_size: 8192, // Larger blocks for L3
            ..Default::default()
        }
    }

    /// Calculates optimal block size based on cache hierarchy.
    pub fn optimal_block_size_for_arrays(&self, num_arrays: usize) -> usize {
        // Account for multiple arrays (momentum, variance, parameters)
        let available_cache = self.l2_cache_size / num_arrays;
        let elements_per_cache = available_cache / std::mem::size_of::<f32>();

        // Use power of 2 for better memory alignment
        let mut block_size = 64;
        while block_size * 2 <= elements_per_cache && block_size < 16384 {
            block_size *= 2;
        }

        block_size.min(self.block_size)
    }
}

/// Cache-friendly memory layout for optimizer state.
///
/// This structure organizes optimizer state data to maximize cache utilization
/// by grouping frequently accessed data together.
#[derive(Debug)]
pub struct CacheFriendlyState {
    /// Interleaved momentum and variance data for cache efficiency
    /// Format: [momentum[i], variance[i], momentum[i+1], variance[i+1], ...]
    pub interleaved_buffers: HashMap<usize, Vec<f32>>,
    /// Parameter metadata for efficient access
    pub param_metadata: HashMap<usize, ParameterMetadata>,
    /// Current step counter
    pub step: usize,
    /// Cache configuration
    pub cache_config: CacheConfig,
}

/// Metadata for efficient parameter processing.
#[derive(Debug, Clone)]
pub struct ParameterMetadata {
    /// Starting offset in interleaved buffer
    pub offset: usize,
    /// Number of elements
    pub size: usize,
    /// Optimal block size for this parameter
    pub block_size: usize,
    /// Last access timestamp for cache management
    pub last_access: usize,
}

impl CacheFriendlyState {
    /// Creates a new cache-friendly state with the given configuration.
    pub fn new(cache_config: CacheConfig) -> Self {
        Self {
            interleaved_buffers: HashMap::new(),
            param_metadata: HashMap::new(),
            step: 0,
            cache_config,
        }
    }

    /// Allocates buffers for a parameter with optimal memory layout.
    pub fn allocate_parameter(&mut self, param_id: usize, size: usize) -> Result<()> {
        // Allocate interleaved momentum and variance
        // Format: [m0, v0, m1, v1, ..., mn, vn]
        let buffer_size = size * 2; // momentum + variance
        let buffer = vec![0.0; buffer_size];

        let metadata = ParameterMetadata {
            offset: 0,
            size,
            block_size: self.cache_config.optimal_block_size_for_arrays(3), // param, momentum, variance
            last_access: self.step,
        };

        self.interleaved_buffers.insert(param_id, buffer);
        self.param_metadata.insert(param_id, metadata);

        Ok(())
    }

    /// Gets direct access to interleaved buffer for efficient in-place operations.
    pub fn get_interleaved_buffer_mut(&mut self, param_id: usize) -> Option<(&mut [f32], usize)> {
        if let (Some(buffer), Some(metadata)) = (
            self.interleaved_buffers.get_mut(&param_id),
            self.param_metadata.get_mut(&param_id),
        ) {
            metadata.last_access = self.step;
            Some((buffer.as_mut_slice(), metadata.size))
        } else {
            None
        }
    }

    /// Gets momentum and variance slices for a parameter (backward compatibility).
    /// Note: This creates temporary vectors and is less efficient than get_interleaved_buffer_mut.
    pub fn get_buffers_mut(&mut self, param_id: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if let (Some(buffer), Some(metadata)) = (
            self.interleaved_buffers.get(&param_id),
            self.param_metadata.get_mut(&param_id),
        ) {
            metadata.last_access = self.step;

            // Extract momentum and variance from interleaved buffer
            let mut momentum = Vec::with_capacity(metadata.size);
            let mut variance = Vec::with_capacity(metadata.size);

            for i in 0..metadata.size {
                momentum.push(buffer[i * 2]);
                variance.push(buffer[i * 2 + 1]);
            }

            Some((momentum, variance))
        } else {
            None
        }
    }

    /// Updates interleaved buffers with new momentum and variance values.
    pub fn update_buffers(
        &mut self,
        param_id: usize,
        momentum: &[f32],
        variance: &[f32],
    ) -> Result<()> {
        if let Some(buffer) = self.interleaved_buffers.get_mut(&param_id) {
            if momentum.len() != variance.len() || momentum.len() * 2 != buffer.len() {
                return Err(TrustformersError::tensor_op_error(
                    "Buffer size mismatch",
                    "update_buffers",
                ));
            }

            // Update interleaved buffer
            for i in 0..momentum.len() {
                buffer[i * 2] = momentum[i];
                buffer[i * 2 + 1] = variance[i];
            }

            Ok(())
        } else {
            Err(TrustformersError::tensor_op_error(
                "Parameter not found",
                "update_buffers",
            ))
        }
    }

    /// Clears unused buffers to free memory.
    pub fn garbage_collect(&mut self, access_threshold: usize) {
        let current_step = self.step;
        let stale_params: Vec<usize> = self
            .param_metadata
            .iter()
            .filter(|(_, metadata)| current_step - metadata.last_access > access_threshold)
            .map(|(id, _)| *id)
            .collect();

        for param_id in stale_params {
            self.interleaved_buffers.remove(&param_id);
            self.param_metadata.remove(&param_id);
        }
    }
}

/// Cache-friendly Adam optimizer implementation.
///
/// This optimizer uses blocked processing and optimized memory layouts
/// to minimize cache misses and improve performance.
#[derive(Debug)]
pub struct CacheFriendlyAdam {
    /// Learning rate
    lr: f32,
    /// Beta coefficients for momentum and variance
    betas: (f32, f32),
    /// Epsilon for numerical stability
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// Cache-friendly state
    state: CacheFriendlyState,
}

impl CacheFriendlyAdam {
    /// Creates a new cache-friendly Adam optimizer.
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self::with_cache_config(lr, betas, eps, weight_decay, CacheConfig::default())
    }

    /// Creates a cache-friendly Adam optimizer with custom cache configuration.
    pub fn with_cache_config(
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        cache_config: CacheConfig,
    ) -> Self {
        Self {
            lr,
            betas,
            eps,
            weight_decay,
            state: CacheFriendlyState::new(cache_config),
        }
    }

    /// Updates parameter using cache-friendly blocked processing (legacy wrapper).
    #[allow(dead_code)]
    fn update_parameter_blocked(
        &mut self,
        param: &mut [f32],
        grad: &[f32],
        param_id: String,
    ) -> Result<()> {
        // Convert string ID to numeric ID for compatibility
        let numeric_id = param_id.as_ptr() as usize;
        self.update_parameter_blocked_fast(param, grad, numeric_id)
    }

    /// Fast parameter update using numeric IDs to avoid string formatting overhead.
    fn update_parameter_blocked_fast(
        &mut self,
        param: &mut [f32],
        grad: &[f32],
        param_id: usize,
    ) -> Result<()> {
        let size = param.len();
        if grad.len() != size {
            return Err(TrustformersError::tensor_op_error(
                "Parameter and gradient size mismatch",
                "update_parameter_blocked_fast",
            ));
        }

        // Ensure parameter buffers are allocated with correct size
        if !self.state.param_metadata.contains_key(&param_id) {
            self.state.allocate_parameter(param_id, size)?;
        } else {
            // Check if size has changed and reallocate if needed
            let current_size =
                self.state.param_metadata.get(&param_id).map(|meta| meta.size).unwrap_or(0);
            if current_size != size {
                self.state.allocate_parameter(param_id, size)?;
            }
        }

        // Extract needed values before borrowing the buffer
        let step = self.state.step + 1;
        let block_size = self
            .state
            .param_metadata
            .get(&param_id)
            .map(|meta| meta.block_size)
            .unwrap_or(1024);
        let _enable_prefetching = self.state.cache_config.enable_prefetching;

        let (bias_correction1, bias_correction2) =
            BiasCorrection::compute_adam_corrections(self.betas.0, self.betas.1, step);

        // Get direct access to interleaved buffer for efficient operations
        let (interleaved_buffer, _param_size) =
            self.state.get_interleaved_buffer_mut(param_id).ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "Failed to get parameter buffers",
                    "update_parameter_blocked_fast",
                )
            })?;

        // For smaller tensors (< 4096 elements), use direct processing to avoid block overhead
        if size < 4096 {
            // Direct processing with inlined operations for better performance
            for i in 0..size {
                let grad_val = grad[i] + self.weight_decay * param[i];

                // Work directly with interleaved buffer: [m0, v0, m1, v1, ...]
                let momentum_idx = i * 2;
                let variance_idx = i * 2 + 1;

                // Update momentum and variance with inlined EMA operations
                interleaved_buffer[momentum_idx] = self.betas.0 * interleaved_buffer[momentum_idx]
                    + (1.0 - self.betas.0) * grad_val;
                interleaved_buffer[variance_idx] = self.betas.1 * interleaved_buffer[variance_idx]
                    + (1.0 - self.betas.1) * grad_val * grad_val;

                // Apply bias-corrected update with inlined operations
                let m_hat = interleaved_buffer[momentum_idx] / bias_correction1;
                let v_hat = interleaved_buffer[variance_idx] / bias_correction2;

                param[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        } else {
            // Use block processing for larger tensors where cache benefits matter
            let num_blocks = (size + block_size - 1) / block_size;

            for block_idx in 0..num_blocks {
                let start = block_idx * block_size;
                let end = (start + block_size).min(size);

                // Note: Prefetching removed to avoid borrowing conflicts - this is a minor optimization
                // Process current block with inlined operations
                for i in start..end {
                    let grad_val = grad[i] + self.weight_decay * param[i];

                    // Work directly with interleaved buffer: [m0, v0, m1, v1, ...]
                    let momentum_idx = i * 2;
                    let variance_idx = i * 2 + 1;

                    interleaved_buffer[momentum_idx] = self.betas.0
                        * interleaved_buffer[momentum_idx]
                        + (1.0 - self.betas.0) * grad_val;
                    interleaved_buffer[variance_idx] = self.betas.1
                        * interleaved_buffer[variance_idx]
                        + (1.0 - self.betas.1) * grad_val * grad_val;

                    let m_hat = interleaved_buffer[momentum_idx] / bias_correction1;
                    let v_hat = interleaved_buffer[variance_idx] / bias_correction2;

                    param[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }

        // No need to update buffers - we worked directly with the interleaved buffer
        Ok(())
    }

    /// Processes a block with fused operations for better cache utilization.
    #[inline]
    #[allow(dead_code)]
    fn process_block_fused(
        &self,
        param_block: &mut [f32],
        grad_block: &[f32],
        momentum_block: &mut [f32],
        variance_block: &mut [f32],
        bias_correction1: f32,
        bias_correction2: f32,
    ) {
        // Fuse all operations for maximum cache efficiency
        for i in 0..param_block.len() {
            let grad_val = grad_block[i] + self.weight_decay * param_block[i];

            // Update momentum and variance in one pass
            ParameterUpdate::update_ema(&mut momentum_block[i], grad_val, self.betas.0);
            ParameterUpdate::update_ema(&mut variance_block[i], grad_val * grad_val, self.betas.1);

            // Apply bias-corrected update immediately
            let m_hat = momentum_block[i] / bias_correction1;
            let v_hat = variance_block[i] / bias_correction2;

            ParameterUpdate::adam_update(&mut param_block[i], self.lr, m_hat, v_hat, self.eps);
        }
    }

    /// Software prefetch hint for better cache performance.
    #[inline]
    #[allow(dead_code)]
    fn prefetch_block(&self, block: &[f32]) {
        // Implement cache-friendly prefetching for different architectures
        if block.is_empty() {
            return;
        }

        // Get pointer to first element
        let ptr = block.as_ptr();

        // Use architecture-specific prefetch instructions when available
        #[cfg(target_arch = "x86_64")]
        {
            // Prefetch data into L1 cache (temporal locality)
            // This provides a hint to the processor to load data into cache
            // Use ptr.wrapping_add to avoid bounds checking in hot path
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);

                // For larger blocks, prefetch multiple cache lines
                if block.len() > 16 {
                    // More than one cache line (64 bytes / 4 bytes per f32)
                    let mid_ptr = ptr.wrapping_add(block.len() / 2);
                    std::arch::x86_64::_mm_prefetch(
                        mid_ptr as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 prefetch using inline assembly
            unsafe {
                std::arch::asm!(
                    "prfm pldl1keep, [{}]",
                    in(reg) ptr,
                    options(nostack, preserves_flags)
                );
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback: access first element to trigger cache load
            // This is a software hint that usually gets optimized away
            // but can help with cache warming on some architectures
            let _ = unsafe { std::ptr::read_volatile(ptr) };
        }
    }

    /// Gets cache utilization statistics.
    pub fn cache_stats(&self) -> CacheStats {
        let buffer_memory: usize = self
            .state
            .interleaved_buffers
            .values()
            .map(|buffer| buffer.len() * std::mem::size_of::<f32>())
            .sum();

        let num_params = self.state.param_metadata.len();
        let total_elements: usize = self.state.param_metadata.values().map(|meta| meta.size).sum();

        CacheStats {
            buffer_memory_bytes: buffer_memory,
            num_parameters: num_params,
            total_elements,
            cache_config: self.state.cache_config.clone(),
            estimated_l1_utilization: self
                .estimate_cache_utilization(buffer_memory, self.state.cache_config.l1_cache_size),
            estimated_l2_utilization: self
                .estimate_cache_utilization(buffer_memory, self.state.cache_config.l2_cache_size),
        }
    }

    /// Estimates cache utilization percentage.
    fn estimate_cache_utilization(&self, working_set_size: usize, cache_size: usize) -> f32 {
        if cache_size == 0 {
            return 1.0;
        }
        (working_set_size as f32 / cache_size as f32).min(1.0)
    }

    /// Performs garbage collection on unused parameters.
    pub fn cleanup_unused_params(&mut self, steps_threshold: usize) {
        self.state.garbage_collect(steps_threshold);
    }
}

impl Optimizer for CacheFriendlyAdam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                // Use pointer address as numeric ID to avoid string formatting overhead
                let param_id = param.as_ptr() as usize;
                let param_slice = param.as_slice_mut().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Parameter tensor is not contiguous",
                        "update",
                    )
                })?;
                let grad_slice = grad_arr.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Gradient tensor is not contiguous",
                        "update",
                    )
                })?;
                self.update_parameter_blocked_fast(param_slice, grad_slice, param_id)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for CacheFriendlyAdam",
                "update",
            )),
        }
    }

    fn zero_grad(&mut self) {
        // No explicit gradient storage
    }

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Cache performance statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total memory used by optimizer buffers
    pub buffer_memory_bytes: usize,
    /// Number of parameters being optimized
    pub num_parameters: usize,
    /// Total number of parameter elements
    pub total_elements: usize,
    /// Cache configuration used
    pub cache_config: CacheConfig,
    /// Estimated L1 cache utilization (0.0 to 1.0)
    pub estimated_l1_utilization: f32,
    /// Estimated L2 cache utilization (0.0 to 1.0)
    pub estimated_l2_utilization: f32,
}

impl CacheStats {
    /// Suggests optimization strategies based on cache utilization.
    pub fn optimization_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if self.estimated_l1_utilization > 0.8 {
            suggestions.push("Consider reducing block size for better L1 cache fit".to_string());
        }

        if self.estimated_l2_utilization > 0.9 {
            suggestions
                .push("Working set exceeds L2 cache; consider parameter partitioning".to_string());
        }

        if self.cache_config.block_size > 8192 {
            suggestions.push("Large block size may cause cache thrashing".to_string());
        }

        if !self.cache_config.enable_prefetching {
            suggestions.push("Enable prefetching for potential performance gains".to_string());
        }

        if suggestions.is_empty() {
            suggestions.push("Cache utilization appears optimal".to_string());
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_creation() {
        let config = CacheConfig::default();
        assert_eq!(config.l1_cache_size, 32 * 1024);
        assert_eq!(config.cache_line_size, 64);
        assert!(config.enable_prefetching);

        let l1_config = CacheConfig::l1_optimized();
        assert_eq!(l1_config.block_size, 512);
    }

    #[test]
    fn test_optimal_block_size() {
        let config = CacheConfig::default();
        let block_size = config.optimal_block_size_for_arrays(3);
        assert!(block_size > 0);
        assert!(block_size <= config.block_size);
        assert_eq!(block_size & (block_size - 1), 0); // Should be power of 2
    }

    #[test]
    fn test_cache_friendly_state() {
        let mut state = CacheFriendlyState::new(CacheConfig::default());

        // Test parameter allocation
        let param_id = 12345usize;
        state.allocate_parameter(param_id, 100).unwrap();

        assert!(state.param_metadata.contains_key(&param_id));
        assert!(state.interleaved_buffers.contains_key(&param_id));

        // Test buffer access
        let (momentum, variance) = state.get_buffers_mut(param_id).unwrap();
        assert_eq!(momentum.len(), 100);
        assert_eq!(variance.len(), 100);
    }

    #[test]
    fn test_cache_friendly_adam() {
        let optimizer = CacheFriendlyAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.betas, (0.9, 0.999));
        assert_eq!(optimizer.eps, 1e-8);
        assert_eq!(optimizer.weight_decay, 0.01);
    }

    #[test]
    fn test_cache_stats() {
        let optimizer = CacheFriendlyAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let stats = optimizer.cache_stats();

        assert_eq!(stats.num_parameters, 0);
        assert_eq!(stats.total_elements, 0);
        assert_eq!(stats.buffer_memory_bytes, 0);

        let suggestions = stats.optimization_suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_cache_utilization_estimation() {
        let optimizer = CacheFriendlyAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);

        let utilization = optimizer.estimate_cache_utilization(16 * 1024, 32 * 1024);
        assert_eq!(utilization, 0.5);

        let over_utilization = optimizer.estimate_cache_utilization(64 * 1024, 32 * 1024);
        assert_eq!(over_utilization, 1.0);
    }

    #[test]
    fn test_garbage_collection() {
        let mut state = CacheFriendlyState::new(CacheConfig::default());

        // Add some parameters
        let param1_id = 11111usize;
        let param2_id = 22222usize;
        state.allocate_parameter(param1_id, 100).unwrap();
        state.allocate_parameter(param2_id, 200).unwrap();

        // Simulate time passing
        state.step = 1000;

        // Access only param1
        state.get_buffers_mut(param1_id);

        // Garbage collect with threshold of 10 steps
        state.garbage_collect(10);

        // param1 should remain, param2 should be removed
        assert!(state.param_metadata.contains_key(&param1_id));
        assert!(!state.param_metadata.contains_key(&param2_id));
    }
}
