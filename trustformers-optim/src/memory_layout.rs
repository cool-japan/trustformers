//! Memory layout optimizations for improved cache performance.
//!
//! This module provides data structures and algorithms optimized for cache-friendly
//! memory layouts, reducing memory bandwidth usage and improving performance through
//! better spatial and temporal locality.
//!
//! # Key Optimizations
//!
//! - **Structure of Arrays (SoA)**: Better vectorization and cache usage
//! - **Memory Alignment**: Ensure data aligns to cache line boundaries
//! - **Hot/Cold Data Separation**: Keep frequently accessed data together
//! - **Prefetch-Friendly Layouts**: Optimize for hardware prefetchers
//! - **NUMA-Aware Allocation**: Optimize for multi-socket systems

use crate::common::{BiasCorrection, ParameterUpdate};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::{self, NonNull};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Memory alignment configuration for optimal cache performance.
#[derive(Debug, Clone, Copy)]
pub struct AlignmentConfig {
    /// Cache line size (typically 64 bytes)
    pub cache_line_size: usize,
    /// Vector register size (typically 32 bytes for AVX2, 64 for AVX-512)
    pub vector_size: usize,
    /// Page size for large allocations (typically 4KB)
    pub page_size: usize,
    /// Enable huge pages for large allocations
    pub use_huge_pages: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            cache_line_size: 64,
            vector_size: 32, // AVX2
            page_size: 4096,
            use_huge_pages: false,
        }
    }
}

impl AlignmentConfig {
    /// Creates configuration optimized for AVX-512.
    pub fn avx512() -> Self {
        Self {
            vector_size: 64,
            ..Default::default()
        }
    }

    /// Creates configuration with huge pages enabled.
    pub fn with_huge_pages() -> Self {
        Self {
            use_huge_pages: true,
            ..Default::default()
        }
    }

    /// Gets the alignment requirement for the given size.
    pub fn alignment_for_size(&self, size: usize) -> usize {
        if size >= self.page_size {
            self.page_size
        } else if size >= self.cache_line_size {
            self.cache_line_size
        } else {
            self.vector_size.min(size)
        }
    }
}

/// Aligned memory allocator for cache-friendly data structures.
#[derive(Debug)]
pub struct AlignedAllocator {
    config: AlignmentConfig,
    allocated_blocks: Vec<(NonNull<u8>, Layout)>,
}

impl AlignedAllocator {
    /// Creates a new aligned allocator.
    pub fn new(config: AlignmentConfig) -> Self {
        Self {
            config,
            allocated_blocks: Vec::new(),
        }
    }

    /// Allocates aligned memory for the given type and count.
    pub fn allocate_aligned<T>(&mut self, count: usize) -> Result<NonNull<T>> {
        let size = count * std::mem::size_of::<T>();
        let alignment = self.config.alignment_for_size(size);

        let layout = Layout::from_size_align(size, alignment).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Invalid layout: {}", e),
                "allocate_aligned",
            )
        })?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(TrustformersError::tensor_op_error(
                "Memory allocation failed",
                "allocate_aligned",
            ));
        }

        let non_null = NonNull::new(ptr).ok_or_else(|| {
            TrustformersError::tensor_op_error("Null pointer in allocation", "allocate_aligned")
        })?;

        self.allocated_blocks.push((non_null, layout));

        // Cast to the target type
        let typed_ptr = non_null.as_ptr() as *mut T;
        NonNull::new(typed_ptr).ok_or_else(|| {
            TrustformersError::tensor_op_error("Type casting failed", "allocate_aligned")
        })
    }

    /// Allocates and initializes aligned memory.
    pub fn allocate_initialized<T: Clone>(&mut self, count: usize, value: T) -> Result<NonNull<T>> {
        let ptr = self.allocate_aligned::<T>(count)?;

        unsafe {
            for i in 0..count {
                ptr::write(ptr.as_ptr().add(i), value.clone());
            }
        }

        Ok(ptr)
    }

    /// Gets memory usage statistics.
    pub fn memory_usage(&self) -> usize {
        self.allocated_blocks.iter().map(|(_, layout)| layout.size()).sum()
    }
}

impl Drop for AlignedAllocator {
    fn drop(&mut self) {
        for (ptr, layout) in &self.allocated_blocks {
            unsafe {
                dealloc(ptr.as_ptr(), *layout);
            }
        }
    }
}

// Safety: AlignedAllocator manages owned memory allocations properly
// and the NonNull pointers are used as owned memory handles
unsafe impl Send for AlignedAllocator {}
unsafe impl Sync for AlignedAllocator {}

/// Structure of Arrays (SoA) layout for optimizer state.
///
/// Momentum and variance for *every* registered parameter live in two contiguous
/// arrays rather than in per-parameter allocations, so a sweep over one moment
/// touches consecutive cache lines. Each parameter owns the half-open range
/// `[momentum_offset, momentum_offset + size)` of `SoAOptimizerState::momentum`
/// (and likewise for the variance array); ranges are padded up to a cache-line
/// boundary so no two parameters share a cache line.
///
/// The moments are **persistent**: `update_parameter_soa` reads the previous values
/// out of these arrays and writes the new ones back. An earlier revision recomputed
/// both from the current gradient on every call, which silently removed all of
/// Adam's adaptivity.
#[derive(Debug)]
pub struct SoAOptimizerState {
    /// Contiguous first-moment (momentum) storage shared by all parameters.
    momentum: Vec<f32>,
    /// Contiguous second-moment (variance) storage shared by all parameters.
    variance: Vec<f32>,
    /// Parameter metadata
    parameters: Vec<ParameterInfo>,
    /// Fast lookup from parameter id to its index in `parameters`.
    parameter_index: HashMap<String, usize>,
    /// Global step counter
    step: usize,
    /// Alignment configuration
    alignment: AlignmentConfig,
}

/// Information about a parameter in SoA layout.
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Parameter ID
    pub id: String,
    /// Starting index in momentum array
    pub momentum_offset: usize,
    /// Starting index in variance array
    pub variance_offset: usize,
    /// Number of elements
    pub size: usize,
    /// Cache-friendly chunk size
    pub chunk_size: usize,
}

impl SoAOptimizerState {
    /// Creates a new SoA optimizer state.
    pub fn new(alignment: AlignmentConfig) -> Self {
        Self {
            momentum: Vec::new(),
            variance: Vec::new(),
            parameters: Vec::new(),
            parameter_index: HashMap::new(),
            step: 0,
            alignment,
        }
    }

    /// Number of `f32` elements per cache line, used to pad parameter blocks.
    fn cache_line_elements(&self) -> usize {
        (self.alignment.cache_line_size / std::mem::size_of::<f32>()).max(1)
    }

    /// Adds a parameter to the SoA layout, reserving zeroed moment storage for it.
    ///
    /// Registering the same id twice is a no-op, so callers may register lazily.
    ///
    /// # Errors
    ///
    /// Returns an error when `size` is zero, which would produce an empty block.
    pub fn add_parameter(&mut self, id: String, size: usize) -> Result<()> {
        if self.parameter_index.contains_key(&id) {
            return Ok(());
        }
        if size == 0 {
            return Err(TrustformersError::tensor_op_error(
                "cannot register a zero-sized parameter in the SoA layout",
                "add_parameter",
            ));
        }

        // Calculate optimal chunk size for vectorization
        let chunk_size = self.calculate_optimal_chunk_size(size);

        // Each parameter's block starts on a cache-line boundary within the shared
        // array so that neighbouring parameters never share a line.
        let line = self.cache_line_elements();
        let momentum_offset = self.momentum.len().div_ceil(line) * line;
        let variance_offset = self.variance.len().div_ceil(line) * line;

        self.momentum.resize(momentum_offset + size, 0.0);
        self.variance.resize(variance_offset + size, 0.0);

        let param_info = ParameterInfo {
            id: id.clone(),
            momentum_offset,
            variance_offset,
            size,
            chunk_size,
        };

        self.parameter_index.insert(id, self.parameters.len());
        self.parameters.push(param_info);
        Ok(())
    }

    /// Calculates optimal chunk size for vectorization.
    fn calculate_optimal_chunk_size(&self, size: usize) -> usize {
        let vector_elements = self.alignment.vector_size / std::mem::size_of::<f32>();
        let cache_line_elements = self.alignment.cache_line_size / std::mem::size_of::<f32>();

        // Choose chunk size that aligns with both vector and cache line boundaries
        let min_chunk = vector_elements;
        let preferred_chunk = cache_line_elements;

        if size >= preferred_chunk {
            preferred_chunk
        } else if size >= min_chunk {
            // Round down to nearest vector size
            (size / min_chunk) * min_chunk
        } else {
            size
        }
    }

    /// Gets parameter information by ID.
    pub fn get_parameter_info(&self, id: &str) -> Option<&ParameterInfo> {
        self.parameter_index.get(id).and_then(|&i| self.parameters.get(i))
    }

    /// Reads the persisted first moment of a parameter (primarily for tests).
    pub fn momentum_of(&self, id: &str) -> Option<&[f32]> {
        let info = self.get_parameter_info(id)?;
        self.momentum.get(info.momentum_offset..info.momentum_offset + info.size)
    }

    /// Reads the persisted second moment of a parameter (primarily for tests).
    pub fn variance_of(&self, id: &str) -> Option<&[f32]> {
        let info = self.get_parameter_info(id)?;
        self.variance.get(info.variance_offset..info.variance_offset + info.size)
    }

    /// Current global step counter.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Updates momentum and variance for a parameter using optimized memory access.
    pub fn update_parameter_soa(
        &mut self,
        param_id: &str,
        param: &mut [f32],
        grad: &[f32],
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Result<()> {
        let param_info = self
            .get_parameter_info(param_id)
            .ok_or_else(|| {
                TrustformersError::tensor_op_error("Parameter not found", "update_parameter_soa")
            })?
            .clone();

        if param.len() != param_info.size || grad.len() != param_info.size {
            return Err(TrustformersError::tensor_op_error(
                "Size mismatch",
                "update_parameter_soa",
            ));
        }

        self.step += 1;
        let (bias_correction1, bias_correction2) =
            BiasCorrection::compute_adam_corrections(betas.0, betas.1, self.step);

        // Process in cache-friendly chunks
        let chunk_size = param_info.chunk_size;
        let num_chunks = param_info.size.div_ceil(chunk_size);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(param_info.size);

            self.process_chunk_soa(
                &mut param[start..end],
                &grad[start..end],
                start,
                &param_info,
                lr,
                betas,
                bias_correction1,
                bias_correction2,
                eps,
                weight_decay,
            )?;
        }

        Ok(())
    }

    /// Processes a chunk using the Structure of Arrays layout.
    ///
    /// Reads the previous moments out of the shared arrays, applies the Adam EMA
    /// update, writes the new moments back, and steps the parameter.
    fn process_chunk_soa(
        &mut self,
        param_chunk: &mut [f32],
        grad_chunk: &[f32],
        offset: usize,
        param_info: &ParameterInfo,
        lr: f32,
        betas: (f32, f32),
        bias_correction1: f32,
        bias_correction2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<()> {
        let momentum_start = param_info.momentum_offset + offset;
        let variance_start = param_info.variance_offset + offset;
        let len = param_chunk.len();

        // `momentum` and `variance` are distinct fields, so both slices can be held
        // mutably at once.
        let momentum_slice =
            self.momentum.get_mut(momentum_start..momentum_start + len).ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "momentum block out of range for SoA layout",
                    "process_chunk_soa",
                )
            })?;
        let variance_slice =
            self.variance.get_mut(variance_start..variance_start + len).ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "variance block out of range for SoA layout",
                    "process_chunk_soa",
                )
            })?;

        for i in 0..len {
            let grad_val = grad_chunk[i] + weight_decay * param_chunk[i];

            // Real SoA access: load the persisted moments for this element.
            let momentum = &mut momentum_slice[i];
            let variance = &mut variance_slice[i];

            // Update momentum and variance with exponential moving averages
            ParameterUpdate::update_ema(momentum, grad_val, betas.0);
            ParameterUpdate::update_ema(variance, grad_val * grad_val, betas.1);

            // Compute bias-corrected estimates
            let m_hat = *momentum / bias_correction1;
            let v_hat = *variance / bias_correction2;

            // Apply Adam update to parameter
            ParameterUpdate::adam_update(&mut param_chunk[i], lr, m_hat, v_hat, eps);
        }

        Ok(())
    }

    /// Gets memory layout statistics.
    pub fn layout_stats(&self) -> LayoutStats {
        let momentum_memory = self.momentum.len() * std::mem::size_of::<f32>();
        let variance_memory = self.variance.len() * std::mem::size_of::<f32>();
        let total_elements: usize = self.parameters.iter().map(|p| p.size).sum();

        LayoutStats {
            total_parameters: self.parameters.len(),
            total_elements,
            momentum_memory_bytes: momentum_memory,
            variance_memory_bytes: variance_memory,
            total_memory_bytes: momentum_memory + variance_memory,
            alignment_config: self.alignment,
            cache_line_utilization: self.calculate_cache_line_utilization(),
        }
    }

    /// Calculates cache line utilization efficiency.
    fn calculate_cache_line_utilization(&self) -> f32 {
        if self.parameters.is_empty() {
            return 1.0;
        }

        let cache_line_elements = self.alignment.cache_line_size / std::mem::size_of::<f32>();
        let mut total_utilization = 0.0;

        for param in &self.parameters {
            let lines_used = param.size.div_ceil(cache_line_elements);
            let elements_in_lines = lines_used * cache_line_elements;
            let utilization = param.size as f32 / elements_in_lines as f32;
            total_utilization += utilization;
        }

        total_utilization / self.parameters.len() as f32
    }
}

// Safety: SoAOptimizerState contains AlignedAllocator which manages memory properly
unsafe impl Send for SoAOptimizerState {}
unsafe impl Sync for SoAOptimizerState {}

/// Memory layout optimization statistics.
#[derive(Debug, Clone)]
pub struct LayoutStats {
    /// Number of parameters
    pub total_parameters: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Memory used by momentum arrays
    pub momentum_memory_bytes: usize,
    /// Memory used by variance arrays
    pub variance_memory_bytes: usize,
    /// Total memory usage
    pub total_memory_bytes: usize,
    /// Alignment configuration
    pub alignment_config: AlignmentConfig,
    /// Cache line utilization efficiency (0.0 to 1.0)
    pub cache_line_utilization: f32,
}

impl LayoutStats {
    /// Calculates memory overhead compared to naive layout.
    pub fn memory_overhead(&self) -> f32 {
        let naive_memory = self.total_elements * std::mem::size_of::<f32>() * 2; // momentum + variance
        if naive_memory == 0 {
            return 0.0;
        }
        (self.total_memory_bytes as f32 / naive_memory as f32) - 1.0
    }

    /// Suggests layout optimizations.
    pub fn optimization_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if self.cache_line_utilization < 0.8 {
            suggestions.push("Poor cache line utilization; consider parameter padding".to_string());
        }

        let overhead = self.memory_overhead();
        if overhead > 0.2 {
            suggestions.push(format!(
                "High memory overhead ({:.1}%); review alignment requirements",
                overhead * 100.0
            ));
        }

        if self.alignment_config.vector_size > 32 && self.total_elements < 1000 {
            suggestions.push("Vector size may be too large for small parameters".to_string());
        }

        if !self.alignment_config.use_huge_pages && self.total_memory_bytes > 1024 * 1024 {
            suggestions.push("Consider enabling huge pages for large memory usage".to_string());
        }

        if suggestions.is_empty() {
            suggestions.push("Memory layout appears well optimized".to_string());
        }

        suggestions
    }
}

/// Memory-optimized Adam optimizer using SoA layout.
#[derive(Debug)]
pub struct LayoutOptimizedAdam {
    /// Learning rate
    lr: f32,
    /// Beta coefficients
    betas: (f32, f32),
    /// Epsilon for numerical stability
    eps: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// SoA optimizer state
    state: SoAOptimizerState,
    /// Stable parameter identity registry (see [`crate::param_id`]).
    ///
    /// Replaces heap-address keys, which change in every process and so made
    /// checkpoint resume silently restore nothing.
    params: crate::param_id::ParamRegistry,
}

impl LayoutOptimizedAdam {
    /// Creates a new layout-optimized Adam optimizer.
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self::with_alignment(lr, betas, eps, weight_decay, AlignmentConfig::default())
    }

    /// Creates an optimizer with custom alignment configuration.
    pub fn with_alignment(
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        alignment: AlignmentConfig,
    ) -> Self {
        Self {
            lr,
            betas,
            eps,
            weight_decay,
            state: SoAOptimizerState::new(alignment),
            params: crate::param_id::ParamRegistry::new(),
        }
    }

    /// Creates an AVX-512 optimized variant.
    pub fn avx512_optimized(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self::with_alignment(lr, betas, eps, weight_decay, AlignmentConfig::avx512())
    }

    /// Gets layout optimization statistics.
    pub fn layout_stats(&self) -> LayoutStats {
        self.state.layout_stats()
    }

    /// Adds a parameter to the optimizer with optimal layout.
    pub fn add_parameter(&mut self, id: String, size: usize) -> Result<()> {
        self.state.add_parameter(id, size)
    }
}

impl Optimizer for LayoutOptimizedAdam {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = self.params.key_for_addr(param.as_ptr() as usize, param.len())?;

                // Ensure parameter is registered
                if self.state.get_parameter_info(&param_id).is_none() {
                    self.state.add_parameter(param_id.clone(), param.len())?;
                }

                let param_slice = param.as_slice_mut().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get mutable slice from param tensor",
                        "update",
                    )
                })?;
                let grad_slice = grad_arr.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get slice from gradient tensor",
                        "update",
                    )
                })?;
                self.state.update_parameter_soa(
                    &param_id,
                    param_slice,
                    grad_slice,
                    self.lr,
                    self.betas,
                    self.eps,
                    self.weight_decay,
                )
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for LayoutOptimizedAdam",
                "update",
            )),
        }
    }

    fn zero_grad(&mut self) {
        // No explicit gradient storage
    }

    fn step(&mut self) {
        // Step counter is handled in update_parameter_soa
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// Safety: LayoutOptimizedAdam contains SoAOptimizerState which is Send/Sync
unsafe impl Send for LayoutOptimizedAdam {}
unsafe impl Sync for LayoutOptimizedAdam {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_config() {
        let config = AlignmentConfig::default();
        assert_eq!(config.cache_line_size, 64);
        assert_eq!(config.vector_size, 32);
        assert!(!config.use_huge_pages);

        let avx512_config = AlignmentConfig::avx512();
        assert_eq!(avx512_config.vector_size, 64);

        let alignment = config.alignment_for_size(1000);
        assert!(alignment > 0);
        assert!(alignment <= config.cache_line_size);
    }

    #[test]
    fn test_aligned_allocator() {
        let config = AlignmentConfig::default();
        let mut allocator = AlignedAllocator::new(config);

        let _ptr = allocator.allocate_aligned::<f32>(1000).expect("Operation failed in test");
        // Pointer is allocated successfully

        let memory_usage = allocator.memory_usage();
        assert!(memory_usage >= 1000 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_soa_optimizer_state() {
        let config = AlignmentConfig::default();
        let mut state = SoAOptimizerState::new(config);

        state
            .add_parameter("param1".to_string(), 1000)
            .expect("Operation failed in test");
        assert!(state.get_parameter_info("param1").is_some());

        let stats = state.layout_stats();
        assert_eq!(stats.total_parameters, 1);
        assert_eq!(stats.total_elements, 1000);
    }

    #[test]
    fn test_layout_optimized_adam() {
        let optimizer = LayoutOptimizedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.betas, (0.9, 0.999));

        let stats = optimizer.layout_stats();
        assert_eq!(stats.total_parameters, 0);
    }

    #[test]
    fn test_layout_stats() {
        let config = AlignmentConfig::default();
        let mut state = SoAOptimizerState::new(config);

        state
            .add_parameter("param1".to_string(), 100)
            .expect("Operation failed in test");
        state
            .add_parameter("param2".to_string(), 200)
            .expect("Operation failed in test");

        let stats = state.layout_stats();
        assert_eq!(stats.total_parameters, 2);
        assert_eq!(stats.total_elements, 300);
        assert!(stats.cache_line_utilization > 0.0);
        assert!(stats.cache_line_utilization <= 1.0);

        let overhead = stats.memory_overhead();
        assert!(overhead >= 0.0);

        let suggestions = stats.optimization_suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_chunk_size_calculation() {
        let config = AlignmentConfig::default();
        let state = SoAOptimizerState::new(config);

        let chunk_size_large = state.calculate_optimal_chunk_size(10000);
        let chunk_size_small = state.calculate_optimal_chunk_size(5);

        assert!(chunk_size_large > chunk_size_small);
        assert!(
            chunk_size_large.is_multiple_of(config.vector_size / std::mem::size_of::<f32>())
                || chunk_size_large == 10000
        );
    }

    #[test]
    fn test_avx512_optimization() {
        let optimizer = LayoutOptimizedAdam::avx512_optimized(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let stats = optimizer.layout_stats();
        assert_eq!(stats.alignment_config.vector_size, 64);
    }

    /// Regression: `process_chunk_soa` used to derive momentum/variance from the
    /// current gradient on every call and never write them back, so the "Adam"
    /// update was a pure function of the latest gradient.
    #[test]
    fn test_soa_state_persists_across_steps() {
        let mut state = SoAOptimizerState::new(AlignmentConfig::default());
        state.add_parameter("w".to_string(), 4).expect("register");

        let mut param = vec![0.0_f32; 4];
        let grad = vec![1.0_f32; 4];

        state
            .update_parameter_soa("w", &mut param, &grad, 0.1, (0.9, 0.999), 1e-8, 0.0)
            .expect("step 1");
        let momentum_after_first = state.momentum_of("w").expect("momentum block").to_vec();
        assert!(
            momentum_after_first.iter().all(|m| (m - 0.1).abs() < 1e-6),
            "first EMA must be (1-beta1)*g = 0.1, got {momentum_after_first:?}"
        );

        // A second step with a ZERO gradient can only move the parameter if the
        // momentum from step 1 was actually stored.
        let before_second = param.clone();
        let zero_grad = vec![0.0_f32; 4];
        state
            .update_parameter_soa("w", &mut param, &zero_grad, 0.1, (0.9, 0.999), 1e-8, 0.0)
            .expect("step 2");

        let momentum_after_second = state.momentum_of("w").expect("momentum block").to_vec();
        assert!(
            momentum_after_second.iter().all(|m| (m - 0.09).abs() < 1e-6),
            "second EMA must decay to beta1*0.1 = 0.09, got {momentum_after_second:?}"
        );
        for (before, after) in before_second.iter().zip(param.iter()) {
            assert!(
                (before - after).abs() > 1e-6,
                "carried momentum must still move the parameter on a zero gradient"
            );
        }
    }

    /// Two parameters must own disjoint, non-overlapping blocks of the shared arrays.
    #[test]
    fn test_soa_parameters_get_disjoint_blocks() {
        let mut state = SoAOptimizerState::new(AlignmentConfig::default());
        state.add_parameter("a".to_string(), 4).expect("a");
        state.add_parameter("b".to_string(), 4).expect("b");

        let mut param_a = vec![0.0_f32; 4];
        let mut param_b = vec![0.0_f32; 4];
        let grad_a = vec![1.0_f32; 4];
        let grad_b = vec![0.0_f32; 4];

        state
            .update_parameter_soa("a", &mut param_a, &grad_a, 0.1, (0.9, 0.999), 1e-8, 0.0)
            .expect("update a");
        state
            .update_parameter_soa("b", &mut param_b, &grad_b, 0.1, (0.9, 0.999), 1e-8, 0.0)
            .expect("update b");

        let momentum_b = state.momentum_of("b").expect("b momentum");
        assert!(
            momentum_b.iter().all(|m| m.abs() < 1e-9),
            "parameter b saw a zero gradient; its momentum must stay zero: {momentum_b:?}"
        );
        let momentum_a = state.momentum_of("a").expect("a momentum");
        assert!(momentum_a.iter().all(|m| *m > 0.0), "a must have momentum");
    }

    /// Convergence smoke test on a quadratic bowl f(x) = sum(x^2), grad = 2x.
    #[test]
    fn test_layout_optimized_adam_converges_on_quadratic() {
        let mut optimizer = LayoutOptimizedAdam::new(0.05, (0.9, 0.999), 1e-8, 0.0);
        let mut param = Tensor::from_vec(vec![1.0_f32; 4], &[4]).expect("param");
        let initial_loss: f32 = param.data().expect("data").iter().map(|v| v * v).sum();

        for _ in 0..400 {
            let grad_data: Vec<f32> = param.data().expect("data").iter().map(|v| 2.0 * v).collect();
            let grad = Tensor::from_vec(grad_data, &[4]).expect("grad");
            optimizer.update(&mut param, &grad).expect("update");
        }

        let final_loss: f32 = param.data().expect("data").iter().map(|v| v * v).sum();
        assert!(
            final_loss < initial_loss * 1e-2,
            "loss must decrease: {initial_loss} -> {final_loss}"
        );
    }
}
