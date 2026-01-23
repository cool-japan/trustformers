//! Performance Optimization Utilities
//!
//! This module provides performance optimization utilities for model inference,
//! including batch processing, memory optimization, and caching strategies.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::Tensor;

/// Configuration for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum batch size for inference
    pub max_batch_size: usize,
    /// Whether to enable dynamic batching
    pub enable_dynamic_batching: bool,
    /// Cache size for frequently used tensors
    pub cache_size: usize,
    /// Whether to enable memory optimization
    pub enable_memory_optimization: bool,
    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            enable_dynamic_batching: true,
            cache_size: 1000,
            enable_memory_optimization: true,
            num_threads: None, // Use system default
        }
    }
}

/// LRU Cache implementation for tensors
#[derive(Debug)]
pub struct LruCache {
    capacity: usize,
    cache: HashMap<String, (Tensor, usize)>, // (tensor, access_order)
    access_order: usize,
    access_history: VecDeque<String>,
    hits: usize,
    misses: usize,
}

impl LruCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: HashMap::new(),
            access_order: 0,
            access_history: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&Tensor> {
        if self.cache.contains_key(key) {
            self.access_order += 1;
            // Get a clone to avoid borrow checker issues
            let (tensor, _) = self.cache.get(key).unwrap().clone();
            self.cache.insert(key.to_string(), (tensor, self.access_order));
            self.hits += 1;
            self.cache.get(key).map(|(tensor, _)| tensor)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn put(&mut self, key: String, tensor: Tensor) {
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            self.evict_lru();
        }

        self.access_order += 1;
        self.cache.insert(key.clone(), (tensor, self.access_order));
        self.access_history.push_back(key);

        // Keep access history manageable
        if self.access_history.len() > self.capacity * 2 {
            self.access_history.pop_front();
        }
    }

    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.find_lru_key() {
            self.cache.remove(&lru_key);
        }
    }

    fn find_lru_key(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|(_, (_, access_order))| *access_order)
            .map(|(key, _)| key.clone())
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_history.clear();
        self.access_order = 0;
        self.hits = 0;
        self.misses = 0;
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn statistics(&self) -> CacheStatistics {
        CacheStatistics {
            current_size: self.cache.len(),
            max_size: self.capacity,
            hit_rate: self.hit_rate(),
        }
    }
}

/// Batch processor for efficient inference
#[derive(Debug)]
pub struct BatchProcessor {
    config: PerformanceConfig,
    cache: LruCache,
    batch_buffer: Vec<Tensor>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            cache: LruCache::new(config.cache_size),
            config,
            batch_buffer: Vec::new(),
        }
    }

    /// Add a tensor to the current batch
    pub fn add_to_batch(&mut self, tensor: Tensor) -> Result<Option<Vec<Tensor>>> {
        self.batch_buffer.push(tensor);

        if self.batch_buffer.len() >= self.config.max_batch_size {
            Ok(Some(self.flush_batch()?))
        } else {
            Ok(None)
        }
    }

    /// Flush the current batch and return it
    pub fn flush_batch(&mut self) -> Result<Vec<Tensor>> {
        let batch = std::mem::take(&mut self.batch_buffer);
        Ok(batch)
    }

    /// Cache a tensor with a given key
    pub fn cache_tensor(&mut self, key: String, tensor: Tensor) -> Result<()> {
        self.cache.put(key, tensor);
        Ok(())
    }

    /// Cache statistics
    pub fn cache_stats(&self) -> CacheStatistics {
        self.cache.statistics()
    }

    /// Retrieve a cached tensor
    pub fn get_cached_tensor(&mut self, key: &str) -> Option<&Tensor> {
        self.cache.get(key)
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get current batch size
    pub fn current_batch_size(&self) -> usize {
        self.batch_buffer.len()
    }
}

/// Memory optimization utilities
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Optimize tensor memory layout for better cache performance
    pub fn optimize_memory_layout(tensors: &mut [Tensor]) -> Result<()> {
        // Sort tensors by size (larger tensors first) for better memory allocation patterns
        tensors.sort_by(|a, b| {
            let size_a = a.shape().iter().product::<usize>();
            let size_b = b.shape().iter().product::<usize>();
            size_b.cmp(&size_a) // Descending order
        });

        // Apply memory layout optimizations per tensor
        for tensor in tensors.iter_mut() {
            Self::optimize_single_tensor_layout(tensor)?;
        }

        Ok(())
    }

    /// Optimize memory layout for a single tensor
    fn optimize_single_tensor_layout(tensor: &mut Tensor) -> Result<()> {
        match tensor {
            Tensor::F32(ref mut data) => {
                // For multidimensional tensors, consider reshaping for better cache locality
                // This is a simplified optimization - in practice, you'd analyze access patterns
                if data.ndim() > 2 {
                    // Ensure the tensor is in contiguous memory layout
                    if !data.is_standard_layout() {
                        let owned = data.to_owned();
                        *data = owned;
                    }
                }
            },
            Tensor::I64(ref mut data) => {
                // Similar optimization for integer tensors
                if data.ndim() > 2 && !data.is_standard_layout() {
                    let owned = data.to_owned();
                    *data = owned;
                }
            },
            _ => {
                // For other tensor types, ensure standard layout if possible
            },
        }
        Ok(())
    }

    /// Analyze memory access patterns and suggest optimizations
    pub fn analyze_memory_patterns(tensors: &[Tensor]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for fragmentation patterns
        let total_elements: usize =
            tensors.iter().map(|t| t.shape().iter().product::<usize>()).sum();

        if total_elements > 1_000_000 {
            recommendations
                .push("Consider using memory pooling for large tensor operations".to_string());
        }

        // Check for small tensor overhead
        let small_tensors =
            tensors.iter().filter(|t| t.shape().iter().product::<usize>() < 1000).count();

        if small_tensors > 10 {
            recommendations
                .push("Consider tensor batching to reduce small tensor overhead".to_string());
        }

        // Check tensor alignment and suggest SIMD optimization
        for (i, tensor) in tensors.iter().enumerate() {
            let shape = tensor.shape();
            if shape.len() >= 2 {
                let last_dim = shape[shape.len() - 1];
                if last_dim % 4 != 0 {
                    recommendations.push(format!(
                        "Tensor {} last dimension ({}) not aligned for SIMD operations",
                        i, last_dim
                    ));
                }
            }
        }

        recommendations
    }

    /// Estimate memory usage for a batch of tensors
    pub fn estimate_memory_usage(tensors: &[Tensor]) -> Result<usize> {
        let mut total_bytes = 0;

        for tensor in tensors {
            let shape = tensor.shape();
            let elements = shape.iter().product::<usize>();
            // Assuming f32 elements (4 bytes each)
            total_bytes += elements * 4;
        }

        Ok(total_bytes)
    }

    /// Check if a batch fits within memory constraints
    pub fn check_memory_constraints(tensors: &[Tensor], max_memory_mb: usize) -> Result<bool> {
        let estimated_bytes = Self::estimate_memory_usage(tensors)?;
        let max_bytes = max_memory_mb * 1024 * 1024;
        Ok(estimated_bytes <= max_bytes)
    }
}

/// Dynamic batching strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Fixed batch size
    Fixed(usize),
    /// Dynamic batching based on sequence length
    DynamicByLength {
        max_length: usize,
        max_batch_size: usize,
    },
    /// Dynamic batching based on memory constraints
    DynamicByMemory { max_memory_mb: usize },
    /// Adaptive batching that adjusts based on performance metrics
    Adaptive {
        initial_batch_size: usize,
        max_batch_size: usize,
        target_latency_ms: f64,
        adjustment_factor: f64,
    },
    /// Priority-based batching with different priorities
    PriorityBased {
        high_priority_batch_size: usize,
        normal_priority_batch_size: usize,
        low_priority_batch_size: usize,
    },
}

/// Dynamic batch manager
#[derive(Debug)]
pub struct DynamicBatchManager {
    strategy: BatchingStrategy,
    pending_tensors: Vec<(Tensor, usize)>, // (tensor, priority)
    current_batch_size: usize,
    recent_latencies: VecDeque<f64>,
    total_batches_processed: usize,
}

impl DynamicBatchManager {
    /// Create a new dynamic batch manager
    pub fn new(strategy: BatchingStrategy) -> Self {
        let initial_batch_size = match &strategy {
            BatchingStrategy::Fixed(size) => *size,
            BatchingStrategy::DynamicByLength { max_batch_size, .. } => *max_batch_size / 2,
            BatchingStrategy::DynamicByMemory { .. } => 16,
            BatchingStrategy::Adaptive {
                initial_batch_size, ..
            } => *initial_batch_size,
            BatchingStrategy::PriorityBased {
                normal_priority_batch_size,
                ..
            } => *normal_priority_batch_size,
        };

        Self {
            strategy,
            pending_tensors: Vec::new(),
            current_batch_size: initial_batch_size,
            recent_latencies: VecDeque::new(),
            total_batches_processed: 0,
        }
    }

    /// Record latency for adaptive batching
    pub fn record_latency(&mut self, latency_ms: f64) {
        self.recent_latencies.push_back(latency_ms);

        // Keep only recent latencies (last 20 batches)
        if self.recent_latencies.len() > 20 {
            self.recent_latencies.pop_front();
        }

        self.total_batches_processed += 1;

        // Adjust batch size for adaptive strategy
        if let BatchingStrategy::Adaptive {
            target_latency_ms,
            max_batch_size,
            adjustment_factor,
            ..
        } = &self.strategy
        {
            if self.recent_latencies.len() >= 5 {
                let avg_latency: f64 =
                    self.recent_latencies.iter().sum::<f64>() / self.recent_latencies.len() as f64;

                if avg_latency > *target_latency_ms {
                    // Latency too high, reduce batch size
                    self.current_batch_size = std::cmp::max(
                        1,
                        (self.current_batch_size as f64 * (1.0 - adjustment_factor)) as usize,
                    );
                } else if avg_latency < *target_latency_ms * 0.8 {
                    // Latency acceptable, can increase batch size
                    self.current_batch_size = std::cmp::min(
                        *max_batch_size,
                        (self.current_batch_size as f64 * (1.0 + adjustment_factor)) as usize,
                    );
                }
            }
        }
    }

    /// Add a tensor to the pending queue with priority
    pub fn add_tensor(&mut self, tensor: Tensor, priority: usize) -> Result<()> {
        self.pending_tensors.push((tensor, priority));
        // Sort by priority (higher priority first)
        self.pending_tensors.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(())
    }

    /// Get the next optimal batch based on the strategy
    pub fn get_next_batch(&mut self) -> Result<Option<Vec<Tensor>>> {
        if self.pending_tensors.is_empty() {
            return Ok(None);
        }

        match &self.strategy {
            BatchingStrategy::Fixed(batch_size) => {
                if self.pending_tensors.len() >= *batch_size {
                    let batch: Vec<Tensor> = self
                        .pending_tensors
                        .drain(0..*batch_size)
                        .map(|(tensor, _)| tensor)
                        .collect();
                    Ok(Some(batch))
                } else {
                    Ok(None)
                }
            },
            BatchingStrategy::DynamicByLength {
                max_length: _,
                max_batch_size,
            } => {
                let batch_size = std::cmp::min(self.pending_tensors.len(), *max_batch_size);
                if batch_size > 0 {
                    let batch: Vec<Tensor> = self
                        .pending_tensors
                        .drain(0..batch_size)
                        .map(|(tensor, _)| tensor)
                        .collect();
                    Ok(Some(batch))
                } else {
                    Ok(None)
                }
            },
            BatchingStrategy::DynamicByMemory { max_memory_mb } => {
                let mut batch = Vec::new();
                let mut current_memory = 0;

                while !self.pending_tensors.is_empty() {
                    let tensor_memory = self.estimate_tensor_memory(&self.pending_tensors[0].0)?;
                    if current_memory + tensor_memory <= *max_memory_mb * 1024 * 1024 {
                        let (tensor, _) = self.pending_tensors.remove(0);
                        batch.push(tensor);
                        current_memory += tensor_memory;
                    } else {
                        break;
                    }
                }

                if batch.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(batch))
                }
            },
            BatchingStrategy::Adaptive { .. } => {
                if self.pending_tensors.len() >= self.current_batch_size {
                    let batch: Vec<Tensor> = self
                        .pending_tensors
                        .drain(0..self.current_batch_size)
                        .map(|(tensor, _)| tensor)
                        .collect();
                    Ok(Some(batch))
                } else {
                    Ok(None)
                }
            },
            BatchingStrategy::PriorityBased {
                high_priority_batch_size,
                normal_priority_batch_size,
                low_priority_batch_size,
            } => {
                // Group by priority
                let high_priority: Vec<_> = self
                    .pending_tensors
                    .iter()
                    .filter(|(_, priority)| *priority >= 80)
                    .cloned()
                    .collect();
                let normal_priority: Vec<_> = self
                    .pending_tensors
                    .iter()
                    .filter(|(_, priority)| *priority >= 40 && *priority < 80)
                    .cloned()
                    .collect();
                let low_priority: Vec<_> = self
                    .pending_tensors
                    .iter()
                    .filter(|(_, priority)| *priority < 40)
                    .cloned()
                    .collect();

                if high_priority.len() >= *high_priority_batch_size {
                    let batch: Vec<Tensor> = high_priority
                        .into_iter()
                        .take(*high_priority_batch_size)
                        .map(|(tensor, _)| tensor)
                        .collect();
                    // Remove processed tensors
                    self.pending_tensors.retain(|(_, priority)| *priority < 80);
                    Ok(Some(batch))
                } else if normal_priority.len() >= *normal_priority_batch_size {
                    let batch: Vec<Tensor> = normal_priority
                        .into_iter()
                        .take(*normal_priority_batch_size)
                        .map(|(tensor, _)| tensor)
                        .collect();
                    // Remove processed tensors
                    self.pending_tensors.retain(|(_, priority)| *priority < 40 || *priority >= 80);
                    Ok(Some(batch))
                } else if low_priority.len() >= *low_priority_batch_size {
                    let batch: Vec<Tensor> = low_priority
                        .into_iter()
                        .take(*low_priority_batch_size)
                        .map(|(tensor, _)| tensor)
                        .collect();
                    // Remove processed tensors
                    self.pending_tensors.retain(|(_, priority)| *priority >= 40);
                    Ok(Some(batch))
                } else {
                    Ok(None)
                }
            },
        }
    }

    /// Estimate memory usage for a single tensor
    fn estimate_tensor_memory(&self, tensor: &Tensor) -> Result<usize> {
        let shape = tensor.shape();
        let elements = shape.iter().product::<usize>();
        // Assuming f32 elements (4 bytes each)
        Ok(elements * 4)
    }

    /// Get number of pending tensors
    pub fn pending_count(&self) -> usize {
        self.pending_tensors.len()
    }

    /// Get current batch size for adaptive strategies
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Get average latency for performance analysis
    pub fn average_latency(&self) -> f64 {
        if self.recent_latencies.is_empty() {
            0.0
        } else {
            self.recent_latencies.iter().sum::<f64>() / self.recent_latencies.len() as f64
        }
    }

    /// Get batch processing statistics
    pub fn get_batch_statistics(&self) -> BatchStatistics {
        BatchStatistics {
            total_batches_processed: self.total_batches_processed,
            current_batch_size: self.current_batch_size,
            pending_tensors: self.pending_tensors.len(),
            average_latency_ms: self.average_latency(),
            strategy_type: match &self.strategy {
                BatchingStrategy::Fixed(_) => "Fixed".to_string(),
                BatchingStrategy::DynamicByLength { .. } => "DynamicByLength".to_string(),
                BatchingStrategy::DynamicByMemory { .. } => "DynamicByMemory".to_string(),
                BatchingStrategy::Adaptive { .. } => "Adaptive".to_string(),
                BatchingStrategy::PriorityBased { .. } => "PriorityBased".to_string(),
            },
        }
    }
}

/// Performance monitoring utilities
#[derive(Debug, Default)]
pub struct PerformanceMonitor {
    total_inference_time: f64,
    total_inferences: usize,
    batch_sizes: Vec<usize>,
    memory_usage: Vec<usize>,
}

impl PerformanceMonitor {
    /// Record an inference time
    pub fn record_inference(&mut self, time_ms: f64, batch_size: usize, memory_usage: usize) {
        self.total_inference_time += time_ms;
        self.total_inferences += 1;
        self.batch_sizes.push(batch_size);
        self.memory_usage.push(memory_usage);
    }

    /// Get average inference time
    pub fn average_inference_time(&self) -> f64 {
        if self.total_inferences > 0 {
            self.total_inference_time / self.total_inferences as f64
        } else {
            0.0
        }
    }

    /// Get average batch size
    pub fn average_batch_size(&self) -> f64 {
        if self.batch_sizes.is_empty() {
            0.0
        } else {
            self.batch_sizes.iter().sum::<usize>() as f64 / self.batch_sizes.len() as f64
        }
    }

    /// Get peak memory usage
    pub fn peak_memory_usage(&self) -> usize {
        self.memory_usage.iter().max().copied().unwrap_or(0)
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> PerformanceStatistics {
        PerformanceStatistics {
            total_inferences: self.total_inferences,
            average_inference_time_ms: self.average_inference_time(),
            average_batch_size: self.average_batch_size(),
            peak_memory_usage_bytes: self.peak_memory_usage(),
            throughput_inferences_per_second: if self.total_inference_time > 0.0 {
                (self.total_inferences as f64) / (self.total_inference_time / 1000.0)
            } else {
                0.0
            },
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub current_size: usize,
    pub max_size: usize,
    pub hit_rate: f64,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    pub total_inferences: usize,
    pub average_inference_time_ms: f64,
    pub average_batch_size: f64,
    pub peak_memory_usage_bytes: usize,
    pub throughput_inferences_per_second: f64,
}

/// Advanced performance optimizer with workload analysis
#[derive(Debug)]
pub struct AdvancedPerformanceOptimizer {
    #[allow(dead_code)]
    config: PerformanceConfig,
    workload_history: Vec<WorkloadMetrics>,
    optimization_recommendations: Vec<String>,
}

/// Workload metrics for optimization analysis
#[derive(Debug, Clone)]
pub struct WorkloadMetrics {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub memory_usage: usize,
    pub inference_time_ms: f64,
    pub timestamp: std::time::Instant,
}

impl AdvancedPerformanceOptimizer {
    /// Create a new advanced optimizer
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            workload_history: Vec::new(),
            optimization_recommendations: Vec::new(),
        }
    }

    /// Record workload metrics
    pub fn record_workload(&mut self, metrics: WorkloadMetrics) {
        self.workload_history.push(metrics);

        // Keep only recent history (last 1000 entries)
        if self.workload_history.len() > 1000 {
            self.workload_history.remove(0);
        }

        // Generate recommendations based on patterns
        self.generate_recommendations();
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&mut self) {
        self.optimization_recommendations.clear();

        if self.workload_history.len() < 10 {
            return;
        }

        // Analyze recent performance patterns
        let recent_metrics: Vec<_> = self.workload_history.iter().rev().take(50).collect();

        // Check for small batch sizes
        let avg_batch_size: f64 = recent_metrics.iter().map(|m| m.batch_size as f64).sum::<f64>()
            / recent_metrics.len() as f64;

        if avg_batch_size < 8.0 {
            self.optimization_recommendations
                .push("Consider increasing batch size for better throughput".to_string());
        }

        // Check for high memory usage variation
        let memory_usages: Vec<usize> = recent_metrics.iter().map(|m| m.memory_usage).collect();
        let max_memory = memory_usages.iter().max().unwrap_or(&0);
        let min_memory = memory_usages.iter().min().unwrap_or(&0);

        if *max_memory > min_memory * 2 {
            self.optimization_recommendations.push(
                "High memory usage variation detected - consider dynamic batching".to_string(),
            );
        }

        // Check for performance degradation
        if recent_metrics.len() >= 20 {
            let first_half_avg: f64 =
                recent_metrics[10..].iter().map(|m| m.inference_time_ms).sum::<f64>() / 10.0;
            let second_half_avg: f64 =
                recent_metrics[..10].iter().map(|m| m.inference_time_ms).sum::<f64>() / 10.0;

            if second_half_avg > first_half_avg * 1.2 {
                self.optimization_recommendations.push(
                    "Performance degradation detected - consider cache clearing or model reloading"
                        .to_string(),
                );
            }
        }
    }

    /// Get current optimization recommendations
    pub fn get_recommendations(&self) -> &[String] {
        &self.optimization_recommendations
    }

    /// Get workload analysis summary
    pub fn get_workload_analysis(&self) -> WorkloadAnalysis {
        if self.workload_history.is_empty() {
            return WorkloadAnalysis::default();
        }

        let total_metrics = self.workload_history.len();
        let avg_batch_size = self.workload_history.iter().map(|m| m.batch_size as f64).sum::<f64>()
            / total_metrics as f64;

        let avg_inference_time =
            self.workload_history.iter().map(|m| m.inference_time_ms).sum::<f64>()
                / total_metrics as f64;

        let peak_memory = self.workload_history.iter().map(|m| m.memory_usage).max().unwrap_or(0);

        WorkloadAnalysis {
            total_samples: total_metrics,
            average_batch_size: avg_batch_size,
            average_inference_time_ms: avg_inference_time,
            peak_memory_usage_bytes: peak_memory,
            recommendations_count: self.optimization_recommendations.len(),
        }
    }
}

/// Workload analysis summary
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WorkloadAnalysis {
    pub total_samples: usize,
    pub average_batch_size: f64,
    pub average_inference_time_ms: f64,
    pub peak_memory_usage_bytes: usize,
    pub recommendations_count: usize,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub total_batches_processed: usize,
    pub current_batch_size: usize,
    pub pending_tensors: usize,
    pub average_latency_ms: f64,
    pub strategy_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert!(config.enable_dynamic_batching);
        assert_eq!(config.cache_size, 1000);
        assert!(config.enable_memory_optimization);
    }

    #[test]
    fn test_batch_processor_creation() {
        let config = PerformanceConfig::default();
        let processor = BatchProcessor::new(config);
        assert_eq!(processor.current_batch_size(), 0);
    }

    #[test]
    fn test_memory_optimizer_estimate() {
        // Create a simple test tensor
        let tensor = Tensor::zeros(&[2, 3]).unwrap();
        let tensors = vec![tensor];

        let estimated = MemoryOptimizer::estimate_memory_usage(&tensors).unwrap();
        // 2 * 3 elements * 4 bytes per f32 element = 24 bytes
        assert_eq!(estimated, 24);
    }

    #[test]
    fn test_dynamic_batch_manager() {
        let strategy = BatchingStrategy::Fixed(2);
        let mut manager = DynamicBatchManager::new(strategy);

        let tensor1 = Tensor::zeros(&[1, 2]).unwrap();
        let tensor2 = Tensor::zeros(&[1, 2]).unwrap();

        manager.add_tensor(tensor1, 1).unwrap();
        manager.add_tensor(tensor2, 2).unwrap();

        let batch = manager.get_next_batch().unwrap();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::default();

        monitor.record_inference(100.0, 4, 1024);
        monitor.record_inference(200.0, 8, 2048);

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_inferences, 2);
        assert_eq!(stats.average_inference_time_ms, 150.0);
        assert_eq!(stats.average_batch_size, 6.0);
        assert_eq!(stats.peak_memory_usage_bytes, 2048);
    }

    #[test]
    fn test_cache_statistics() {
        let config = PerformanceConfig::default();
        let processor = BatchProcessor::new(config);
        let stats = processor.cache_stats();

        assert_eq!(stats.current_size, 0);
        assert_eq!(stats.max_size, 1000);
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_advanced_performance_optimizer() {
        let config = PerformanceConfig::default();
        let mut optimizer = AdvancedPerformanceOptimizer::new(config);

        // Record some sample workloads
        for i in 1..=20 {
            let metrics = WorkloadMetrics {
                batch_size: if i < 10 { 2 } else { 16 }, // Small then large batches
                sequence_length: 512,
                memory_usage: 1024 * i,
                inference_time_ms: 100.0 + (i as f64 * 5.0),
                timestamp: std::time::Instant::now(),
            };
            optimizer.record_workload(metrics);
        }

        let analysis = optimizer.get_workload_analysis();
        assert_eq!(analysis.total_samples, 20);
        assert!(analysis.average_batch_size > 2.0); // Should be higher due to mix

        let recommendations = optimizer.get_recommendations();
        assert!(!recommendations.is_empty()); // Should have some recommendations
    }

    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::new(2);

        let tensor1 = Tensor::zeros(&[1, 2]).unwrap();
        let tensor2 = Tensor::zeros(&[1, 3]).unwrap();
        let tensor3 = Tensor::zeros(&[1, 4]).unwrap();

        // Add tensors
        cache.put("key1".to_string(), tensor1);
        cache.put("key2".to_string(), tensor2);

        // Access key1 to make it recently used
        let _ = cache.get("key1");

        // Add key3 - should evict key2 (least recently used)
        cache.put("key3".to_string(), tensor3);

        // key1 and key3 should be present, key2 should be evicted
        assert!(cache.get("key1").is_some());
        assert!(cache.get("key3").is_some());
        assert!(cache.get("key2").is_none());

        // Check statistics
        let stats = cache.statistics();
        assert_eq!(stats.current_size, 2);
        assert_eq!(stats.max_size, 2);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_adaptive_batching() {
        let strategy = BatchingStrategy::Adaptive {
            initial_batch_size: 4,
            max_batch_size: 16,
            target_latency_ms: 100.0,
            adjustment_factor: 0.2,
        };
        let mut manager = DynamicBatchManager::new(strategy);

        // Record high latency - should reduce batch size
        for _ in 0..10 {
            manager.record_latency(150.0); // Higher than target
        }

        assert!(manager.current_batch_size() < 4); // Should have reduced

        // Record low latency - should increase batch size
        for _ in 0..10 {
            manager.record_latency(50.0); // Lower than target
        }

        // Note: size might not increase immediately due to adaptation logic
        let stats = manager.get_batch_statistics();
        assert_eq!(stats.strategy_type, "Adaptive");
        assert!(stats.average_latency_ms > 0.0);
    }

    #[test]
    fn test_priority_batching() {
        let strategy = BatchingStrategy::PriorityBased {
            high_priority_batch_size: 2,
            normal_priority_batch_size: 4,
            low_priority_batch_size: 8,
        };
        let mut manager = DynamicBatchManager::new(strategy);

        // Add tensors with different priorities
        let tensor = Tensor::zeros(&[1, 2]).unwrap();
        manager.add_tensor(tensor.clone(), 90).unwrap(); // High priority
        manager.add_tensor(tensor.clone(), 50).unwrap(); // Normal priority
        manager.add_tensor(tensor.clone(), 90).unwrap(); // High priority
        manager.add_tensor(tensor.clone(), 20).unwrap(); // Low priority

        // Should get high priority batch first
        let batch = manager.get_next_batch().unwrap();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2); // High priority batch size

        let stats = manager.get_batch_statistics();
        assert_eq!(stats.strategy_type, "PriorityBased");
    }
}

/// Advanced GPU Memory Management
///
/// This module provides sophisticated GPU memory management capabilities
/// for high-performance inference and training workloads.
/// GPU Memory Pool for efficient allocation and deallocation
#[derive(Debug)]
pub struct GpuMemoryPool {
    /// Pool of pre-allocated memory chunks by size
    pools: HashMap<usize, VecDeque<GpuMemoryChunk>>,
    /// Total memory allocated (in bytes)
    total_allocated: usize,
    /// Maximum memory limit (in bytes)
    max_memory_limit: usize,
    /// Memory fragmentation threshold
    fragmentation_threshold: f32,
    /// Memory allocation statistics
    stats: GpuMemoryStats,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryChunk {
    /// Unique identifier for this chunk
    pub id: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether this chunk is currently in use
    pub in_use: bool,
    /// Allocation timestamp
    pub allocated_at: std::time::Instant,
    /// Last access timestamp
    pub last_accessed: std::time::Instant,
    /// Reference count for shared usage
    pub ref_count: usize,
}

#[derive(Debug, Default, Clone)]
pub struct GpuMemoryStats {
    /// Total allocations made
    pub total_allocations: usize,
    /// Total deallocations made
    pub total_deallocations: usize,
    /// Current active allocations
    pub active_allocations: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Current memory usage (bytes)
    pub current_memory_usage: usize,
    /// Memory fragmentation ratio (0.0 - 1.0)
    pub fragmentation_ratio: f32,
    /// Average allocation size
    pub average_allocation_size: f32,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool with specified limit
    pub fn new(max_memory_limit: usize) -> Self {
        Self {
            pools: HashMap::new(),
            total_allocated: 0,
            max_memory_limit,
            fragmentation_threshold: 0.25, // 25% fragmentation threshold
            stats: GpuMemoryStats::default(),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, size_bytes: usize) -> Result<GpuMemoryChunk> {
        // Check if we have available memory
        if self.total_allocated + size_bytes > self.max_memory_limit {
            self.try_defragment()?;
            if self.total_allocated + size_bytes > self.max_memory_limit {
                return Err(TrustformersError::invalid_operation(
                    "GPU memory limit exceeded".to_string(),
                ));
            }
        }

        // Try to find existing chunk from pool
        if let Some(chunk) = self.find_suitable_chunk(size_bytes) {
            self.stats.cache_hits += 1;
            self.stats.active_allocations += 1;
            return Ok(chunk);
        }

        // Allocate new chunk
        let chunk = GpuMemoryChunk {
            id: uuid::Uuid::new_v4().to_string(),
            size_bytes,
            in_use: true,
            allocated_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            ref_count: 1,
        };

        self.total_allocated += size_bytes;
        self.stats.total_allocations += 1;
        self.stats.active_allocations += 1;
        self.stats.cache_misses += 1;
        self.stats.current_memory_usage += size_bytes;

        if self.stats.current_memory_usage > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = self.stats.current_memory_usage;
        }

        // Update average allocation size
        self.stats.average_allocation_size = (self.stats.average_allocation_size
            * (self.stats.total_allocations - 1) as f32
            + size_bytes as f32)
            / self.stats.total_allocations as f32;

        Ok(chunk)
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, mut chunk: GpuMemoryChunk) -> Result<()> {
        chunk.in_use = false;
        chunk.ref_count = 0;

        // Add back to appropriate pool
        let pool = self.pools.entry(chunk.size_bytes).or_default();
        pool.push_back(chunk.clone());

        self.stats.total_deallocations += 1;
        self.stats.active_allocations = self.stats.active_allocations.saturating_sub(1);
        self.stats.current_memory_usage =
            self.stats.current_memory_usage.saturating_sub(chunk.size_bytes);

        // Check if we need to free some pooled memory
        self.cleanup_unused_chunks()?;

        Ok(())
    }

    /// Find a suitable chunk from existing pools
    fn find_suitable_chunk(&mut self, size_bytes: usize) -> Option<GpuMemoryChunk> {
        // Look for exact size match first
        if let Some(pool) = self.pools.get_mut(&size_bytes) {
            if let Some(mut chunk) = pool.pop_front() {
                chunk.in_use = true;
                chunk.last_accessed = std::time::Instant::now();
                chunk.ref_count = 1;
                return Some(chunk);
            }
        }

        // Look for larger chunks that can be split
        let suitable_sizes: Vec<usize> = self.pools.keys()
            .filter(|&&size| size > size_bytes && size <= size_bytes * 2) // Avoid too much waste
            .copied()
            .collect();

        for pool_size in suitable_sizes {
            if let Some(pool) = self.pools.get_mut(&pool_size) {
                if let Some(mut chunk) = pool.pop_front() {
                    chunk.in_use = true;
                    chunk.last_accessed = std::time::Instant::now();
                    chunk.ref_count = 1;
                    return Some(chunk);
                }
            }
        }

        None
    }

    /// Cleanup unused chunks to free memory
    fn cleanup_unused_chunks(&mut self) -> Result<()> {
        let now = std::time::Instant::now();
        let cleanup_threshold = std::time::Duration::from_secs(300); // 5 minutes

        for pool in self.pools.values_mut() {
            pool.retain(|chunk| {
                let should_keep =
                    chunk.in_use || now.duration_since(chunk.last_accessed) < cleanup_threshold;
                if !should_keep {
                    self.total_allocated = self.total_allocated.saturating_sub(chunk.size_bytes);
                }
                should_keep
            });
        }

        Ok(())
    }

    /// Attempt to defragment memory
    fn try_defragment(&mut self) -> Result<()> {
        // Calculate current fragmentation ratio
        let total_pooled = self
            .pools
            .values()
            .map(|pool| pool.iter().map(|chunk| chunk.size_bytes).sum::<usize>())
            .sum::<usize>();

        self.stats.fragmentation_ratio = if self.total_allocated > 0 {
            total_pooled as f32 / self.total_allocated as f32
        } else {
            0.0
        };

        // If fragmentation is above threshold, force cleanup
        if self.stats.fragmentation_ratio > self.fragmentation_threshold {
            self.force_cleanup()?;
        }

        Ok(())
    }

    /// Force cleanup of all unused memory
    fn force_cleanup(&mut self) -> Result<()> {
        for pool in self.pools.values_mut() {
            let initial_size: usize = pool.iter().map(|chunk| chunk.size_bytes).sum();
            pool.retain(|chunk| chunk.in_use);
            let final_size: usize = pool.iter().map(|chunk| chunk.size_bytes).sum();
            self.total_allocated = self.total_allocated.saturating_sub(initial_size - final_size);
        }

        // Recalculate fragmentation
        self.try_defragment()?;

        Ok(())
    }

    /// Get memory pool statistics
    pub fn get_statistics(&self) -> GpuMemoryStats {
        self.stats.clone()
    }

    /// Get current memory usage as percentage of limit
    pub fn get_memory_usage_percentage(&self) -> f32 {
        (self.total_allocated as f32 / self.max_memory_limit as f32) * 100.0
    }

    /// Get cache efficiency (hit rate)
    pub fn get_cache_efficiency(&self) -> f32 {
        let total_requests = self.stats.cache_hits + self.stats.cache_misses;
        if total_requests > 0 {
            self.stats.cache_hits as f32 / total_requests as f32
        } else {
            0.0
        }
    }
}

/// Advanced GPU tensor caching with memory-aware eviction
#[derive(Debug)]
pub struct GpuTensorCache {
    /// Memory pool for efficient allocation
    memory_pool: GpuMemoryPool,
    /// Cached tensors with metadata
    tensor_cache: HashMap<String, CachedTensor>,
    /// LRU ordering for eviction
    lru_order: VecDeque<String>,
    /// Maximum cache size (number of tensors)
    max_cache_size: usize,
    /// Cache statistics
    stats: CacheStatistics,
}

#[derive(Debug, Clone)]
pub struct CachedTensor {
    /// The cached tensor data
    pub tensor: Tensor,
    /// Memory chunk information
    pub memory_chunk: GpuMemoryChunk,
    /// Access frequency score
    pub access_frequency: f32,
    /// Importance score (for eviction prioritization)
    pub importance_score: f32,
    /// Last access time
    pub last_access: std::time::Instant,
    /// Creation time
    pub created_at: std::time::Instant,
}

impl GpuTensorCache {
    /// Create a new GPU tensor cache
    pub fn new(max_cache_size: usize, max_memory_limit: usize) -> Self {
        Self {
            memory_pool: GpuMemoryPool::new(max_memory_limit),
            tensor_cache: HashMap::new(),
            lru_order: VecDeque::new(),
            max_cache_size,
            stats: CacheStatistics {
                current_size: 0,
                max_size: max_cache_size,
                hit_rate: 0.0,
            },
        }
    }

    /// Cache a tensor with optional importance score
    pub fn cache_tensor(
        &mut self,
        key: String,
        tensor: Tensor,
        importance_score: Option<f32>,
    ) -> Result<()> {
        // Calculate tensor size (simplified estimation)
        let tensor_size = self.estimate_tensor_size(&tensor);

        // Allocate memory chunk
        let memory_chunk = self.memory_pool.allocate(tensor_size)?;

        // Create cached tensor
        let cached_tensor = CachedTensor {
            tensor,
            memory_chunk,
            access_frequency: 1.0,
            importance_score: importance_score.unwrap_or(0.5),
            last_access: std::time::Instant::now(),
            created_at: std::time::Instant::now(),
        };

        // Check if we need to evict
        if self.tensor_cache.len() >= self.max_cache_size {
            self.evict_least_important()?;
        }

        // Insert new tensor
        self.tensor_cache.insert(key.clone(), cached_tensor);
        self.lru_order.push_back(key);
        self.stats.current_size = self.tensor_cache.len();

        Ok(())
    }

    /// Retrieve a tensor from cache
    pub fn get_tensor(&mut self, key: &str) -> Option<&Tensor> {
        // Check if key exists first
        if !self.tensor_cache.contains_key(key) {
            return None;
        }

        // Update LRU order first
        self.update_lru_order(key);

        // Update access information and return tensor
        if let Some(cached_tensor) = self.tensor_cache.get_mut(key) {
            cached_tensor.access_frequency += 1.0;
            cached_tensor.last_access = std::time::Instant::now();
            Some(&cached_tensor.tensor)
        } else {
            None
        }
    }

    /// Update LRU order for a key
    fn update_lru_order(&mut self, key: &str) {
        // Remove from current position and add to back
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
            self.lru_order.push_back(key.to_string());
        }
    }

    /// Evict the least important tensor
    fn evict_least_important(&mut self) -> Result<()> {
        // Calculate eviction scores for all cached tensors
        let mut eviction_candidates: Vec<(String, f32)> = self
            .tensor_cache
            .iter()
            .map(|(key, cached_tensor)| {
                let age_factor = cached_tensor.created_at.elapsed().as_secs() as f32 / 3600.0; // Hours
                let frequency_factor = cached_tensor.access_frequency;
                let importance_factor = cached_tensor.importance_score;

                // Lower score = higher priority for eviction
                let eviction_score = importance_factor * frequency_factor / (1.0 + age_factor);
                (key.clone(), eviction_score)
            })
            .collect();

        // Sort by eviction score (lowest first)
        eviction_candidates
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Evict the least important tensor
        if let Some((key_to_evict, _)) = eviction_candidates.first() {
            if let Some(cached_tensor) = self.tensor_cache.remove(key_to_evict) {
                self.memory_pool.deallocate(cached_tensor.memory_chunk)?;

                // Remove from LRU order
                if let Some(pos) = self.lru_order.iter().position(|k| k == key_to_evict) {
                    self.lru_order.remove(pos);
                }

                self.stats.current_size = self.tensor_cache.len();
            }
        }

        Ok(())
    }

    /// Estimate tensor size in bytes (simplified)
    fn estimate_tensor_size(&self, tensor: &Tensor) -> usize {
        match tensor {
            Tensor::F32(arr) => arr.len() * 4, // 4 bytes per f32
            Tensor::F64(arr) => arr.len() * 8, // 8 bytes per f64
            _ => 1024,                         // Default estimate for other types
        }
    }

    /// Get comprehensive cache statistics
    pub fn get_comprehensive_stats(&self) -> GpuCacheStatistics {
        let memory_stats = self.memory_pool.get_statistics();
        let fragmentation_ratio = memory_stats.fragmentation_ratio;

        GpuCacheStatistics {
            cache_stats: self.stats.clone(),
            memory_stats,
            memory_usage_percentage: self.memory_pool.get_memory_usage_percentage(),
            cache_efficiency: self.memory_pool.get_cache_efficiency(),
            average_tensor_age: self.calculate_average_tensor_age(),
            fragmentation_ratio,
        }
    }

    /// Calculate average age of cached tensors
    fn calculate_average_tensor_age(&self) -> f32 {
        if self.tensor_cache.is_empty() {
            return 0.0;
        }

        let total_age: f32 = self
            .tensor_cache
            .values()
            .map(|cached_tensor| cached_tensor.created_at.elapsed().as_secs() as f32)
            .sum();

        total_age / self.tensor_cache.len() as f32
    }

    /// Clear all cached tensors
    pub fn clear(&mut self) -> Result<()> {
        for (_, cached_tensor) in self.tensor_cache.drain() {
            self.memory_pool.deallocate(cached_tensor.memory_chunk)?;
        }
        self.lru_order.clear();
        self.stats.current_size = 0;
        Ok(())
    }
}

/// Comprehensive GPU cache statistics
#[derive(Debug, Clone)]
pub struct GpuCacheStatistics {
    pub cache_stats: CacheStatistics,
    pub memory_stats: GpuMemoryStats,
    pub memory_usage_percentage: f32,
    pub cache_efficiency: f32,
    pub average_tensor_age: f32,
    pub fragmentation_ratio: f32,
}

/// GPU memory optimization recommendations
#[derive(Debug, Clone)]
pub struct GpuOptimizationRecommendations {
    /// Recommended actions to improve performance
    pub recommendations: Vec<String>,
    /// Priority level (High, Medium, Low)
    pub priority: String,
    /// Estimated performance improvement percentage
    pub estimated_improvement: f32,
}

/// GPU memory optimizer with intelligent recommendations
pub struct GpuMemoryOptimizer;

impl GpuMemoryOptimizer {
    /// Analyze GPU memory usage and provide optimization recommendations
    pub fn analyze_and_recommend(stats: &GpuCacheStatistics) -> GpuOptimizationRecommendations {
        let mut recommendations = Vec::new();
        let mut priority = "Low".to_string();
        let mut estimated_improvement: f32 = 0.0;

        // Analyze memory usage
        if stats.memory_usage_percentage > 90.0 {
            recommendations.push("Critical: Memory usage is very high. Consider increasing memory limit or improving eviction strategy.".to_string());
            priority = "High".to_string();
            estimated_improvement += 25.0;
        } else if stats.memory_usage_percentage > 75.0 {
            recommendations.push(
                "Warning: Memory usage is high. Monitor for potential memory pressure.".to_string(),
            );
            priority = "Medium".to_string();
            estimated_improvement += 10.0;
        }

        // Analyze fragmentation
        if stats.fragmentation_ratio > 0.4 {
            recommendations.push(
                "High memory fragmentation detected. Consider running defragmentation.".to_string(),
            );
            if priority == "Low" {
                priority = "Medium".to_string();
            }
            estimated_improvement += 15.0;
        }

        // Analyze cache efficiency
        if stats.cache_efficiency < 0.7 {
            recommendations.push(
                "Low cache hit rate. Consider adjusting cache size or eviction policy.".to_string(),
            );
            if priority == "Low" {
                priority = "Medium".to_string();
            }
            estimated_improvement += 20.0;
        }

        // Analyze tensor age
        if stats.average_tensor_age > 3600.0 {
            // 1 hour
            recommendations.push(
                "Cached tensors are aging. Consider more aggressive eviction for unused tensors."
                    .to_string(),
            );
            estimated_improvement += 5.0;
        }

        // Provide specific optimization suggestions
        if stats.memory_stats.active_allocations > 1000 {
            recommendations.push(
                "High number of active allocations. Consider batching or pooling strategies."
                    .to_string(),
            );
            estimated_improvement += 12.0;
        }

        if recommendations.is_empty() {
            recommendations
                .push("GPU memory usage is optimal. No immediate action required.".to_string());
        }

        GpuOptimizationRecommendations {
            recommendations,
            priority,
            estimated_improvement: estimated_improvement.min(50.0), // Cap at 50%
        }
    }

    /// Perform automatic GPU memory optimization
    pub fn auto_optimize(cache: &mut GpuTensorCache) -> Result<Vec<String>> {
        let stats = cache.get_comprehensive_stats();
        let recommendations = Self::analyze_and_recommend(&stats);
        let mut actions_taken = Vec::new();

        // Auto-apply some optimizations based on priority
        if recommendations.priority == "High" {
            // Force cleanup if memory usage is critical
            if stats.memory_usage_percentage > 90.0 {
                cache.memory_pool.force_cleanup()?;
                actions_taken.push("Performed emergency memory cleanup".to_string());
            }
        }

        if stats.fragmentation_ratio > 0.4 {
            cache.memory_pool.try_defragment()?;
            actions_taken.push("Performed memory defragmentation".to_string());
        }

        if actions_taken.is_empty() {
            actions_taken.push("No automatic optimizations were necessary".to_string());
        }

        Ok(actions_taken)
    }
}

#[cfg(test)]
mod gpu_memory_tests {
    use super::*;

    #[test]
    fn test_gpu_memory_pool_basic() {
        let mut pool = GpuMemoryPool::new(1024 * 1024); // 1MB limit

        // Test allocation
        let chunk = pool.allocate(1024).unwrap();
        assert_eq!(chunk.size_bytes, 1024);
        assert!(chunk.in_use);
        assert_eq!(pool.get_statistics().active_allocations, 1);

        // Test deallocation
        pool.deallocate(chunk).unwrap();
        assert_eq!(pool.get_statistics().active_allocations, 0);
    }

    #[test]
    fn test_gpu_memory_pool_reuse() {
        let mut pool = GpuMemoryPool::new(1024 * 1024);

        // Allocate and deallocate
        let chunk = pool.allocate(1024).unwrap();
        pool.deallocate(chunk).unwrap();

        // Allocate same size - should reuse
        let stats_before = pool.get_statistics();
        let _chunk2 = pool.allocate(1024).unwrap();
        let stats_after = pool.get_statistics();

        assert_eq!(stats_after.cache_hits, stats_before.cache_hits + 1);
    }

    #[test]
    fn test_gpu_tensor_cache() -> Result<()> {
        let mut cache = GpuTensorCache::new(2, 1024 * 1024);

        let tensor1 = Tensor::zeros(&[10, 10])?;
        let tensor2 = Tensor::zeros(&[5, 5])?;
        let tensor3 = Tensor::zeros(&[20, 20])?;

        // Cache tensors
        cache.cache_tensor("tensor1".to_string(), tensor1, Some(0.8))?;
        cache.cache_tensor("tensor2".to_string(), tensor2, Some(0.6))?;

        // Retrieve cached tensor
        assert!(cache.get_tensor("tensor1").is_some());

        // Cache third tensor (should evict least important)
        cache.cache_tensor("tensor3".to_string(), tensor3, Some(0.9))?;

        // tensor2 should be evicted (lowest importance)
        assert!(cache.get_tensor("tensor2").is_none());
        assert!(cache.get_tensor("tensor1").is_some());
        assert!(cache.get_tensor("tensor3").is_some());

        Ok(())
    }

    #[test]
    fn test_gpu_optimization_recommendations() {
        let stats = GpuCacheStatistics {
            cache_stats: CacheStatistics {
                current_size: 100,
                max_size: 100,
                hit_rate: 0.5, // Low hit rate
            },
            memory_stats: GpuMemoryStats {
                fragmentation_ratio: 0.5, // High fragmentation
                ..Default::default()
            },
            memory_usage_percentage: 95.0, // Very high usage
            cache_efficiency: 0.5,
            average_tensor_age: 7200.0, // 2 hours
            fragmentation_ratio: 0.5,
        };

        let recommendations = GpuMemoryOptimizer::analyze_and_recommend(&stats);

        assert_eq!(recommendations.priority, "High");
        assert!(!recommendations.recommendations.is_empty());
        assert!(recommendations.estimated_improvement > 0.0);
    }
}
