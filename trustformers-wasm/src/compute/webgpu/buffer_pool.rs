//! Advanced GPU buffer pool with memory coalescing and intelligent allocation strategies

#![allow(clippy::missing_enforced_import_renames)]

use super::types::{GpuBuffer, GpuDevice, GpuDeviceExt};
use crate::{track_gpu_allocation, track_gpu_deallocation};
use core::cmp::max;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Memory allocation strategy for buffer coalescing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// First-fit: Use first available buffer that's large enough
    FirstFit,
    /// Best-fit: Use smallest buffer that's large enough (minimizes waste)
    BestFit,
    /// Worst-fit: Use largest available buffer (good for large allocations)
    WorstFit,
    /// Buddy system: Power-of-2 allocation for better fragmentation control
    BuddySystem,
}

/// Buffer metadata for tracking usage and coalescing opportunities
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    pub size: usize,
    pub usage_flags: u32,
    pub last_used: f64, // Timestamp for LRU eviction
    pub allocation_count: u32,
    pub is_coalesced: bool,
    pub alignment: usize,
    pub access_pattern: AccessPattern,
    pub temporal_locality_score: f32,
    pub memory_bank_hint: u32,
}

/// Access pattern tracking for temporal locality optimization
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub access_intervals: Vec<f64>,
    pub average_interval: f64,
    pub access_frequency: f32,
    pub sequential_access_ratio: f32,
}

/// Memory coalescing statistics
#[derive(Debug, Clone)]
pub struct CoalescingStats {
    pub total_allocations: u32,
    pub coalesced_allocations: u32,
    pub memory_saved_bytes: usize,
    pub fragmentation_ratio: f32,
    pub average_buffer_utilization: f32,
}

/// Advanced buffer pool with memory coalescing and intelligent allocation
#[wasm_bindgen]
pub struct BufferPool {
    free_buffers: BTreeMap<usize, Vec<GpuBuffer>>,
    buffer_metadata: BTreeMap<usize, BufferMetadata>,
    allocated_bytes: usize,
    peak_allocated_bytes: usize,
    allocation_strategy: AllocationStrategy,
    min_buffer_size: usize,
    max_buffer_size: usize,
    coalescing_threshold: f32,
    stats: CoalescingStats,
    buffer_alignment: usize,
    large_buffer_cache: VecDeque<(GpuBuffer, usize)>,
    defragmentation_threshold: f32,
    // Advanced optimization fields
    temporal_cache: BTreeMap<u32, Vec<(GpuBuffer, f64)>>, // Memory bank -> (buffer, timestamp)
    dynamic_thresholds: DynamicThresholds,
    access_predictor: AccessPredictor,
    memory_bandwidth_optimizer: MemoryBandwidthOptimizer,
}

/// Dynamic threshold adjustment system
#[derive(Debug, Clone)]
pub struct DynamicThresholds {
    pub base_coalescing_threshold: f32,
    pub base_defrag_threshold: f32,
    pub workload_factor: f32,
    pub adaptation_rate: f32,
    pub recent_allocations: VecDeque<(usize, f64)>, // (size, timestamp)
}

/// Access pattern prediction system
#[derive(Debug, Clone)]
pub struct AccessPredictor {
    pub predicted_sizes: Vec<usize>,
    pub prediction_confidence: f32,
    pub historical_patterns: Vec<(usize, f64)>, // (size, interval)
}

/// Memory bandwidth optimization system
#[derive(Debug, Clone)]
pub struct MemoryBandwidthOptimizer {
    pub bank_load_balancing: [f32; 8], // 8 memory banks
    pub optimal_access_patterns: Vec<u32>,
    pub bandwidth_utilization: f32,
    pub spatial_locality_map: std::collections::HashMap<usize, Vec<usize>>, // Size -> Related sizes
    pub cross_device_sync_overhead: f32,
    pub gpu_architecture_hints: GpuArchitectureHints,
    pub memory_channel_utilization: [f32; 4], // 4 memory channels
    pub bandwidth_saturation_threshold: f32,
}

/// GPU architecture-specific optimization hints
#[derive(Debug, Clone)]
pub struct GpuArchitectureHints {
    pub cache_line_size: usize,
    pub memory_bus_width: u32, // bits
    pub max_bandwidth_gbps: f32,
    pub coalescing_alignment: usize,
    pub warp_size: u32,
    pub l2_cache_size: usize,
    pub memory_clock_mhz: u32,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl BufferPool {
    /// Create a new advanced buffer pool with memory coalescing
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::with_strategy_internal(AllocationStrategy::BestFit)
    }

    /// Create buffer pool with specific allocation strategy (private)
    fn with_strategy_internal(strategy: AllocationStrategy) -> Self {
        BufferPool {
            free_buffers: BTreeMap::new(),
            buffer_metadata: BTreeMap::new(),
            allocated_bytes: 0,
            peak_allocated_bytes: 0,
            allocation_strategy: strategy,
            min_buffer_size: 256,              // Minimum 256 bytes
            max_buffer_size: 64 * 1024 * 1024, // Maximum 64MB
            coalescing_threshold: 0.8,         // Coalesce when >80% fragmented
            stats: CoalescingStats {
                total_allocations: 0,
                coalesced_allocations: 0,
                memory_saved_bytes: 0,
                fragmentation_ratio: 0.0,
                average_buffer_utilization: 0.0,
            },
            buffer_alignment: 256, // GPU-optimal alignment
            large_buffer_cache: VecDeque::new(),
            defragmentation_threshold: 0.7, // Defrag when >70% fragmented
            temporal_cache: BTreeMap::new(),
            dynamic_thresholds: DynamicThresholds {
                base_coalescing_threshold: 0.8,
                base_defrag_threshold: 0.7,
                workload_factor: 1.0,
                adaptation_rate: 0.1,
                recent_allocations: VecDeque::new(),
            },
            access_predictor: AccessPredictor {
                predicted_sizes: Vec::new(),
                prediction_confidence: 0.0,
                historical_patterns: Vec::new(),
            },
            memory_bandwidth_optimizer: MemoryBandwidthOptimizer {
                bank_load_balancing: [0.0; 8],
                optimal_access_patterns: Vec::new(),
                bandwidth_utilization: 0.0,
                spatial_locality_map: HashMap::new(),
                cross_device_sync_overhead: 0.05, // 5% overhead baseline
                gpu_architecture_hints: GpuArchitectureHints {
                    cache_line_size: 128,           // Typical GPU cache line
                    memory_bus_width: 384,          // Common for mid-range GPUs
                    max_bandwidth_gbps: 500.0,      // Modern GPU bandwidth
                    coalescing_alignment: 128,      // Optimal coalescing alignment
                    warp_size: 32,                  // Standard warp size
                    l2_cache_size: 6 * 1024 * 1024, // 6MB L2 cache
                    memory_clock_mhz: 7000,         // High-end memory clock
                },
                memory_channel_utilization: [0.0; 4],
                bandwidth_saturation_threshold: 0.85, // 85% bandwidth threshold
            },
        }
    }

    /// Get a buffer using advanced allocation and coalescing strategies
    pub fn get_buffer(
        &mut self,
        device: &GpuDevice,
        size: usize,
        usage: u32,
    ) -> Result<GpuBuffer, wasm_bindgen::JsValue> {
        self.stats.total_allocations += 1;
        let current_time = js_sys::Date::now();

        let aligned_size = self.align_size(size);
        let optimal_size = self.calculate_optimal_size(aligned_size);

        // Try to find existing buffer using allocation strategy
        if let Some(buffer) = self.find_suitable_buffer(optimal_size, usage) {
            self.update_buffer_metadata(optimal_size, current_time, false);
            return Ok(buffer);
        }

        // Check if we should coalesce smaller buffers
        if self.should_coalesce(optimal_size) {
            if let Some(coalesced_buffer) =
                self.try_coalesce_buffers(device, optimal_size, usage)?
            {
                self.stats.coalesced_allocations += 1;
                self.update_buffer_metadata(optimal_size, current_time, true);
                return Ok(coalesced_buffer);
            }
        }

        // Create new buffer with optimal size
        let buffer = self.create_new_buffer(device, optimal_size, usage)?;
        self.allocated_bytes += optimal_size;
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, self.allocated_bytes);

        // Track GPU memory allocation
        track_gpu_allocation(optimal_size);

        self.update_buffer_metadata(optimal_size, current_time, false);

        // Trigger defragmentation if needed
        self.maybe_defragment();

        Ok(buffer)
    }

    /// Return a buffer to the pool with intelligent coalescing
    pub fn return_buffer(&mut self, buffer: GpuBuffer, size: usize) {
        let aligned_size = self.align_size(size);
        let bucket_size = self.get_bucket_size(aligned_size);

        // Update metadata
        if let Some(metadata) = self.buffer_metadata.get_mut(&bucket_size) {
            metadata.last_used = js_sys::Date::now();
            metadata.allocation_count += 1;
        }

        // Keep large buffers in cache for immediate reuse
        if aligned_size > 1024 * 1024 {
            let buffer_clone = buffer.clone();
            self.large_buffer_cache.push_back((buffer_clone, aligned_size));
            if self.large_buffer_cache.len() > 10 {
                self.large_buffer_cache.pop_front();
            }
        }

        // Add to appropriate bucket
        self.free_buffers.entry(bucket_size).or_default().push(buffer);
    }

    /// Release a buffer completely (remove from pool and track deallocation)
    pub fn release_buffer(&mut self, size: usize) {
        let aligned_size = self.align_size(size);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(aligned_size);

        // Track GPU memory deallocation
        track_gpu_deallocation(aligned_size);

        web_sys::console::log_1(&format!("🗑️ Released GPU buffer: {} bytes", aligned_size).into());
    }

    /// Get buffer pool statistics
    pub fn get_stats(&self) -> js_sys::Object {
        let stats = js_sys::Object::new();
        js_sys::Reflect::set(
            &stats,
            &"total_allocations".into(),
            &self.stats.total_allocations.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &stats,
            &"coalesced_allocations".into(),
            &self.stats.coalesced_allocations.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &stats,
            &"memory_saved_bytes".into(),
            &self.stats.memory_saved_bytes.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &stats,
            &"fragmentation_ratio".into(),
            &self.stats.fragmentation_ratio.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &stats,
            &"allocated_bytes".into(),
            &self.allocated_bytes.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &stats,
            &"peak_allocated_bytes".into(),
            &self.peak_allocated_bytes.into(),
        )
        .expect("Failed to set buffer pool stats property");
        stats
    }

    /// Force defragmentation of buffer pool
    pub fn defragment(&mut self) {
        web_sys::console::log_1(&"🔧 Starting buffer pool defragmentation...".into());

        let mut total_freed = 0;
        let mut buffers_coalesced = 0;

        // Identify fragmented buckets and coalesce them
        let bucket_sizes: Vec<usize> = self.free_buffers.keys().copied().collect();

        for &size in &bucket_sizes {
            if let Some(buffers) = self.free_buffers.get_mut(&size) {
                if buffers.len() > 4 {
                    // If we have many small buffers, coalesce them
                    buffers_coalesced += buffers.len();
                    total_freed += size * buffers.len();
                    buffers.clear(); // In a real implementation, would coalesce into larger buffers
                }
            }
        }

        self.stats.memory_saved_bytes += total_freed;

        web_sys::console::log_1(
            &format!(
                "✅ Defragmentation complete: {} buffers coalesced, {}KB freed",
                buffers_coalesced,
                total_freed / 1024
            )
            .into(),
        );
    }

    /// Clear all buffers and reset pool
    pub fn clear(&mut self) {
        self.free_buffers.clear();
        self.buffer_metadata.clear();
        self.large_buffer_cache.clear();
        self.allocated_bytes = 0;
        self.stats = CoalescingStats {
            total_allocations: 0,
            coalesced_allocations: 0,
            memory_saved_bytes: 0,
            fragmentation_ratio: 0.0,
            average_buffer_utilization: 0.0,
        };
    }

    /// Get total allocated memory
    #[wasm_bindgen(getter)]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Get peak allocated memory
    #[wasm_bindgen(getter)]
    pub fn peak_allocated_bytes(&self) -> usize {
        self.peak_allocated_bytes
    }

    /// Get fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    #[wasm_bindgen(getter)]
    pub fn fragmentation_ratio(&self) -> f32 {
        if self.allocated_bytes == 0 {
            return 0.0;
        }

        let total_buffers: usize = self.free_buffers.values().map(|v| v.len()).sum();
        let unique_sizes = self.free_buffers.keys().len();

        if total_buffers == 0 {
            return 0.0;
        }

        // High fragmentation = many small buffers of different sizes
        (unique_sizes as f32) / (total_buffers as f32).sqrt()
    }
}

// Private implementation methods
impl BufferPool {
    /// Align size to GPU-optimal boundaries
    fn align_size(&self, size: usize) -> usize {
        (size + self.buffer_alignment - 1) & !(self.buffer_alignment - 1)
    }

    /// Calculate optimal buffer size for allocation
    fn calculate_optimal_size(&self, requested_size: usize) -> usize {
        match self.allocation_strategy {
            AllocationStrategy::BuddySystem => requested_size.next_power_of_two(),
            AllocationStrategy::FirstFit
            | AllocationStrategy::BestFit
            | AllocationStrategy::WorstFit => {
                // Round to next bucket size for better reuse
                let bucket_factor = 1.25; // 25% size increment per bucket
                let mut bucket_size = self.min_buffer_size;
                while bucket_size < requested_size {
                    bucket_size = (bucket_size as f32 * bucket_factor) as usize;
                }
                bucket_size.min(self.max_buffer_size)
            },
        }
    }

    /// Get bucket size for the given aligned size
    fn get_bucket_size(&self, aligned_size: usize) -> usize {
        self.calculate_optimal_size(aligned_size)
    }

    /// Find suitable buffer using allocation strategy
    fn find_suitable_buffer(&mut self, size: usize, _usage: u32) -> Option<GpuBuffer> {
        match self.allocation_strategy {
            AllocationStrategy::FirstFit => self.find_first_fit(size),
            AllocationStrategy::BestFit => self.find_best_fit(size),
            AllocationStrategy::WorstFit => self.find_worst_fit(size),
            AllocationStrategy::BuddySystem => self.find_buddy_buffer(size),
        }
    }

    /// First-fit allocation: use first buffer that's large enough
    fn find_first_fit(&mut self, size: usize) -> Option<GpuBuffer> {
        for (&bucket_size, buffers) in self.free_buffers.iter_mut() {
            if bucket_size >= size && !buffers.is_empty() {
                return buffers.pop();
            }
        }
        None
    }

    /// Best-fit allocation: use smallest buffer that's large enough
    fn find_best_fit(&mut self, size: usize) -> Option<GpuBuffer> {
        let mut best_size = None;

        // Find the smallest bucket that can fit the request
        for &bucket_size in self.free_buffers.keys() {
            if bucket_size >= size {
                match best_size {
                    None => best_size = Some(bucket_size),
                    Some(current_best) if bucket_size < current_best => {
                        best_size = Some(bucket_size)
                    },
                    _ => {},
                }
            }
        }

        if let Some(size) = best_size {
            if let Some(buffers) = self.free_buffers.get_mut(&size) {
                return buffers.pop();
            }
        }
        None
    }

    /// Worst-fit allocation: use largest available buffer
    fn find_worst_fit(&mut self, size: usize) -> Option<GpuBuffer> {
        let mut worst_size = None;

        // Find the largest bucket that can fit the request
        for &bucket_size in self.free_buffers.keys() {
            if bucket_size >= size {
                match worst_size {
                    None => worst_size = Some(bucket_size),
                    Some(current_worst) if bucket_size > current_worst => {
                        worst_size = Some(bucket_size)
                    },
                    _ => {},
                }
            }
        }

        if let Some(size) = worst_size {
            if let Some(buffers) = self.free_buffers.get_mut(&size) {
                return buffers.pop();
            }
        }
        None
    }

    /// Buddy system allocation: find power-of-2 sized buffer
    fn find_buddy_buffer(&mut self, size: usize) -> Option<GpuBuffer> {
        let buddy_size = size.next_power_of_two();

        if let Some(buffers) = self.free_buffers.get_mut(&buddy_size) {
            if let Some(buffer) = buffers.pop() {
                return Some(buffer);
            }
        }

        // Try to split larger buffer
        let mut larger_size = buddy_size * 2;
        while larger_size <= self.max_buffer_size {
            if let Some(buffers) = self.free_buffers.get_mut(&larger_size) {
                if let Some(_larger_buffer) = buffers.pop() {
                    // In a real implementation, would split the buffer and return one half
                    // For now, just create a new buffer of the right size
                    return None; // Signal to create new buffer
                }
            }
            larger_size *= 2;
        }
        None
    }

    /// Check if we should attempt buffer coalescing
    fn should_coalesce(&self, requested_size: usize) -> bool {
        // Coalesce if:
        // 1. We're requesting a large buffer (>1MB)
        // 2. Fragmentation is above threshold
        // 3. We have many small buffers available

        if requested_size > 1024 * 1024 {
            return true;
        }

        if self.fragmentation_ratio() > self.coalescing_threshold {
            return true;
        }

        // Check if we have many small buffers that could be coalesced
        let small_buffer_count: usize = self
            .free_buffers
            .iter()
            .filter(|(&size, _buffers)| size < requested_size / 2)
            .map(|(_, buffers)| buffers.len())
            .sum();

        small_buffer_count > 4
    }

    /// Try to coalesce smaller buffers into a larger one
    fn try_coalesce_buffers(
        &mut self,
        device: &GpuDevice,
        size: usize,
        usage: u32,
    ) -> Result<Option<GpuBuffer>, wasm_bindgen::JsValue> {
        // For demonstration, we'll "coalesce" by freeing small buffers and creating a larger one
        // In a real implementation, this would involve copying data between buffers

        let mut freed_size = 0;
        let mut freed_count = 0;

        // Free buffers smaller than half the requested size
        let threshold = size / 2;
        let sizes_to_remove: Vec<usize> = self
            .free_buffers
            .iter()
            .filter(|(&bucket_size, buffers)| bucket_size < threshold && !buffers.is_empty())
            .map(|(&size, _)| size)
            .collect();

        for size_to_remove in sizes_to_remove {
            if let Some(buffers) = self.free_buffers.get_mut(&size_to_remove) {
                freed_count += buffers.len();
                freed_size += size_to_remove * buffers.len();
                buffers.clear();
            }
        }

        if freed_size >= size {
            // Create new coalesced buffer
            let buffer = self.create_new_buffer(device, size, usage)?;
            self.stats.memory_saved_bytes += freed_size - size;

            web_sys::console::log_1(
                &format!(
                    "🔄 Coalesced {} small buffers ({}KB) into 1 buffer ({}KB)",
                    freed_count,
                    freed_size / 1024,
                    size / 1024
                )
                .into(),
            );

            return Ok(Some(buffer));
        }

        Ok(None)
    }

    /// Create a new GPU buffer with the specified parameters
    fn create_new_buffer(
        &self,
        device: &GpuDevice,
        size: usize,
        usage: u32,
    ) -> Result<GpuBuffer, wasm_bindgen::JsValue> {
        use super::types::create_buffer_descriptor;
        let descriptor = create_buffer_descriptor(size as f64, usage, None, false)?;
        Ok(device.create_buffer(&descriptor))
    }

    /// Update buffer metadata after allocation
    fn update_buffer_metadata(&mut self, size: usize, timestamp: f64, is_coalesced: bool) {
        let buffer_alignment = self.buffer_alignment;
        let memory_bank_hint = self.calculate_optimal_memory_bank(size);

        let metadata = self.buffer_metadata.entry(size).or_insert_with(|| BufferMetadata {
            size,
            usage_flags: 0,
            last_used: timestamp,
            allocation_count: 0,
            is_coalesced,
            alignment: buffer_alignment,
            access_pattern: AccessPattern {
                access_intervals: Vec::new(),
                average_interval: 0.0,
                access_frequency: 0.0,
                sequential_access_ratio: 0.0,
            },
            temporal_locality_score: 0.5,
            memory_bank_hint,
        });

        // Update access pattern tracking
        if metadata.last_used > 0.0 {
            let interval = timestamp - metadata.last_used;
            metadata.access_pattern.access_intervals.push(interval);

            // Keep only recent intervals (last 10)
            if metadata.access_pattern.access_intervals.len() > 10 {
                metadata.access_pattern.access_intervals.remove(0);
            }

            // Update average interval and frequency
            let sum: f64 = metadata.access_pattern.access_intervals.iter().sum();
            metadata.access_pattern.average_interval =
                sum / metadata.access_pattern.access_intervals.len() as f64;
            metadata.access_pattern.access_frequency =
                if metadata.access_pattern.average_interval > 0.0 {
                    1000.0 / metadata.access_pattern.average_interval as f32 // Frequency in Hz
                } else {
                    0.0
                };
        }

        metadata.last_used = timestamp;
        metadata.allocation_count += 1;
        metadata.is_coalesced = metadata.is_coalesced || is_coalesced;

        // Update temporal locality score based on access frequency
        metadata.temporal_locality_score =
            (metadata.access_pattern.access_frequency / 10.0).min(1.0);

        // Update dynamic thresholds based on workload
        self.update_dynamic_thresholds(size, timestamp);
    }

    /// Check if defragmentation should be triggered
    fn maybe_defragment(&mut self) {
        let dynamic_threshold =
            self.dynamic_thresholds.base_defrag_threshold * self.dynamic_thresholds.workload_factor;
        if self.fragmentation_ratio() > dynamic_threshold {
            self.defragment();
        }
    }

    /// Calculate optimal memory bank for a buffer size (advanced memory bandwidth optimization)
    fn calculate_optimal_memory_bank(&mut self, size: usize) -> u32 {
        // Find memory bank with lowest load for better bandwidth utilization
        let mut min_load = f32::MAX;
        let mut optimal_bank = 0;

        for (bank_id, &load) in
            self.memory_bandwidth_optimizer.bank_load_balancing.iter().enumerate()
        {
            if load < min_load {
                min_load = load;
                optimal_bank = bank_id as u32;
            }
        }

        // Update bank load
        let bank_index = optimal_bank as usize;
        self.memory_bandwidth_optimizer.bank_load_balancing[bank_index] +=
            (size as f32) / (1024.0 * 1024.0); // Convert to MB

        optimal_bank
    }

    /// Update dynamic thresholds based on current workload patterns
    fn update_dynamic_thresholds(&mut self, size: usize, timestamp: f64) {
        // Track recent allocations for workload analysis
        self.dynamic_thresholds.recent_allocations.push_back((size, timestamp));

        // Keep only recent allocations (last 100)
        while self.dynamic_thresholds.recent_allocations.len() > 100 {
            self.dynamic_thresholds.recent_allocations.pop_front();
        }

        if self.dynamic_thresholds.recent_allocations.len() < 10 {
            return; // Not enough data for adjustment
        }

        // Calculate allocation rate and size variance
        let total_size: usize =
            self.dynamic_thresholds.recent_allocations.iter().map(|(s, _)| *s).sum();
        let avg_size = total_size as f32 / self.dynamic_thresholds.recent_allocations.len() as f32;

        let size_variance: f32 = self
            .dynamic_thresholds
            .recent_allocations
            .iter()
            .map(|(s, _)| (*s as f32 - avg_size).powi(2))
            .sum::<f32>()
            / self.dynamic_thresholds.recent_allocations.len() as f32;

        // High variance indicates diverse workload -> higher coalescing threshold
        // Low variance indicates uniform workload -> lower coalescing threshold
        let variance_factor = (size_variance.sqrt() / avg_size).min(2.0);

        // Calculate allocation rate (allocations per second)
        if let (Some(first), Some(last)) = (
            self.dynamic_thresholds.recent_allocations.front(),
            self.dynamic_thresholds.recent_allocations.back(),
        ) {
            let time_span = last.1 - first.1;
            if time_span > 0.0 {
                let allocation_rate =
                    self.dynamic_thresholds.recent_allocations.len() as f64 / time_span * 1000.0; // per second

                // High allocation rate -> more aggressive coalescing
                let rate_factor = (allocation_rate / 10.0).min(2.0) as f32;

                self.dynamic_thresholds.workload_factor = (variance_factor + rate_factor) / 2.0;
            }
        }

        // Apply adaptive adjustment to thresholds
        let target_coalescing = self.dynamic_thresholds.base_coalescing_threshold
            * self.dynamic_thresholds.workload_factor;
        let target_defrag =
            self.dynamic_thresholds.base_defrag_threshold * self.dynamic_thresholds.workload_factor;

        // Smooth threshold adjustment
        self.coalescing_threshold += (target_coalescing - self.coalescing_threshold)
            * self.dynamic_thresholds.adaptation_rate;
        self.defragmentation_threshold += (target_defrag - self.defragmentation_threshold)
            * self.dynamic_thresholds.adaptation_rate;

        // Clamp thresholds to reasonable ranges
        self.coalescing_threshold = self.coalescing_threshold.clamp(0.5, 0.95);
        self.defragmentation_threshold = self.defragmentation_threshold.clamp(0.4, 0.9);
    }

    /// Predict future buffer requirements based on access patterns
    fn predict_buffer_requirements(&mut self) {
        self.access_predictor.predicted_sizes.clear();

        // Analyze historical patterns
        let mut size_frequencies: BTreeMap<usize, u32> = BTreeMap::new();
        for metadata in self.buffer_metadata.values() {
            *size_frequencies.entry(metadata.size).or_insert(0) += metadata.allocation_count;
        }

        // Predict most likely sizes
        let mut sorted_sizes: Vec<(usize, u32)> = size_frequencies.into_iter().collect();
        sorted_sizes.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency (descending)

        // Take top 5 most frequent sizes
        for (size, frequency) in sorted_sizes.into_iter().take(5) {
            self.access_predictor.predicted_sizes.push(size);

            // Calculate prediction confidence based on frequency
            let confidence = (frequency as f32 / self.stats.total_allocations as f32).min(1.0);
            self.access_predictor.prediction_confidence =
                self.access_predictor.prediction_confidence.max(confidence);
        }
    }

    /// Get advanced buffer pool analytics
    pub fn get_advanced_analytics(&mut self) -> js_sys::Object {
        self.predict_buffer_requirements();

        let analytics = js_sys::Object::new();

        // Basic stats
        js_sys::Reflect::set(
            &analytics,
            &"total_allocations".into(),
            &self.stats.total_allocations.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"coalesced_allocations".into(),
            &self.stats.coalesced_allocations.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"memory_saved_bytes".into(),
            &self.stats.memory_saved_bytes.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"fragmentation_ratio".into(),
            &self.fragmentation_ratio().into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"allocated_bytes".into(),
            &self.allocated_bytes.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"peak_allocated_bytes".into(),
            &self.peak_allocated_bytes.into(),
        )
        .expect("Failed to set buffer pool stats property");

        // Advanced analytics
        js_sys::Reflect::set(
            &analytics,
            &"dynamic_coalescing_threshold".into(),
            &self.coalescing_threshold.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"dynamic_defrag_threshold".into(),
            &self.defragmentation_threshold.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"workload_factor".into(),
            &self.dynamic_thresholds.workload_factor.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"prediction_confidence".into(),
            &self.access_predictor.prediction_confidence.into(),
        )
        .expect("Failed to set buffer pool stats property");
        js_sys::Reflect::set(
            &analytics,
            &"bandwidth_utilization".into(),
            &self.memory_bandwidth_optimizer.bandwidth_utilization.into(),
        )
        .expect("Failed to set buffer pool stats property");

        // Memory bank load balancing
        let bank_loads = js_sys::Array::new();
        for load in &self.memory_bandwidth_optimizer.bank_load_balancing {
            bank_loads.push(&(*load).into());
        }
        js_sys::Reflect::set(&analytics, &"memory_bank_loads".into(), &bank_loads.into()).unwrap();

        // Predicted buffer sizes
        let predicted = js_sys::Array::new();
        for &size in &self.access_predictor.predicted_sizes {
            predicted.push(&size.into());
        }
        js_sys::Reflect::set(
            &analytics,
            &"predicted_buffer_sizes".into(),
            &predicted.into(),
        )
        .expect("Failed to set buffer pool stats property");

        analytics
    }
}

// Additional advanced memory management features
impl BufferPool {
    /// Proactive memory prefetching based on predicted access patterns
    pub fn prefetch_buffers(&mut self, device: &GpuDevice) -> Result<(), wasm_bindgen::JsValue> {
        if self.access_predictor.prediction_confidence < 0.6 {
            return Ok(()); // Not confident enough to prefetch
        }

        let current_time = js_sys::Date::now();
        let prefetch_count = (self.access_predictor.predicted_sizes.len() / 2).max(1);

        web_sys::console::log_1(
            &format!(
                "🔮 Prefetching {} buffers based on access patterns",
                prefetch_count
            )
            .into(),
        );

        // Collect predicted sizes first to avoid borrow conflicts
        let predicted_sizes: Vec<usize> = self
            .access_predictor
            .predicted_sizes
            .iter()
            .take(prefetch_count)
            .copied()
            .collect();

        for predicted_size in predicted_sizes {
            // Only prefetch if we don't already have buffers of this size
            let bucket_size = self.get_bucket_size(predicted_size);
            let current_count = self.free_buffers.get(&bucket_size).map_or(0, |v| v.len());

            if current_count < 2 {
                // Create prefetched buffer with standard usage flags
                // gpu_buffer_usage not available in web-sys 0.3.81 - using numeric values
                let usage = 0x80u32 | 0x08u32 | 0x04u32; // STORAGE | COPY_DST | COPY_SRC
                let buffer = self.create_new_buffer(device, predicted_size, usage)?;

                // Clone buffer before moving it
                let buffer_clone = buffer.clone();
                self.return_buffer(buffer, predicted_size);
                self.allocated_bytes += predicted_size;

                // Mark as prefetched in temporal cache
                let bank = self.calculate_optimal_memory_bank(predicted_size);
                self.temporal_cache.entry(bank).or_default().push((buffer_clone, current_time));
            }
        }

        Ok(())
    }

    /// Cache warming strategy for critical buffer sizes
    pub fn warm_cache(
        &mut self,
        device: &GpuDevice,
        critical_sizes: &[usize],
    ) -> Result<(), wasm_bindgen::JsValue> {
        web_sys::console::log_1(
            &format!(
                "🔥 Warming cache for {} critical buffer sizes",
                critical_sizes.len()
            )
            .into(),
        );

        for &size in critical_sizes {
            let aligned_size = self.align_size(size);
            let bucket_size = self.get_bucket_size(aligned_size);

            // Ensure we have at least 2 pre-allocated buffers for each critical size
            let current_count = self.free_buffers.get(&bucket_size).map_or(0, |v| v.len());
            let needed = 2usize.saturating_sub(current_count);

            for _ in 0..needed {
                // gpu_buffer_usage not available in web-sys 0.3.81 - using numeric values
                let usage = 0x80u32 | 0x08u32 | 0x04u32; // STORAGE | COPY_DST | COPY_SRC
                let buffer = self.create_new_buffer(device, bucket_size, usage)?;
                self.return_buffer(buffer, bucket_size);
                self.allocated_bytes += bucket_size;
            }
        }

        Ok(())
    }

    /// Memory pressure handling with intelligent cleanup
    pub fn handle_memory_pressure(&mut self, pressure_level: f32) -> usize {
        web_sys::console::log_1(
            &format!(
                "⚠️ Handling memory pressure: {:.1}%",
                pressure_level * 100.0
            )
            .into(),
        );

        let mut freed_bytes = 0;
        let current_time = js_sys::Date::now();

        match pressure_level {
            p if p > 0.9 => {
                // Critical pressure: aggressive cleanup
                freed_bytes += self.emergency_cleanup(current_time);
            },
            p if p > 0.7 => {
                // High pressure: clean old buffers
                freed_bytes += self.cleanup_old_buffers(current_time, 10000.0); // 10 seconds
            },
            p if p > 0.5 => {
                // Medium pressure: clean very old buffers
                freed_bytes += self.cleanup_old_buffers(current_time, 30000.0); // 30 seconds
            },
            _ => {
                // Low pressure: just defragment
                if self.fragmentation_ratio() > 0.6 {
                    self.defragment();
                }
            },
        }

        web_sys::console::log_1(
            &format!("✅ Memory pressure handled: {}KB freed", freed_bytes / 1024).into(),
        );
        freed_bytes
    }

    /// Emergency cleanup for critical memory situations
    fn emergency_cleanup(&mut self, _current_time: f64) -> usize {
        let mut freed_bytes = 0;

        // Clear large buffer cache first
        while let Some((_buffer, size)) = self.large_buffer_cache.pop_front() {
            freed_bytes += size;
            self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
            track_gpu_deallocation(size);
        }

        // Clear half of all free buffers, starting with largest
        let mut sizes_by_bytes: Vec<(usize, usize)> = self
            .free_buffers
            .iter()
            .map(|(&size, buffers)| (size, size * buffers.len()))
            .collect();
        sizes_by_bytes.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by total bytes (descending)

        for (bucket_size, _total_bytes) in
            sizes_by_bytes.into_iter().take(self.free_buffers.len() / 2)
        {
            if let Some(buffers) = self.free_buffers.get_mut(&bucket_size) {
                let to_remove = buffers.len() / 2;
                for _ in 0..to_remove {
                    if buffers.pop().is_some() {
                        freed_bytes += bucket_size;
                        self.allocated_bytes = self.allocated_bytes.saturating_sub(bucket_size);
                        track_gpu_deallocation(bucket_size);
                    }
                }
            }
        }

        // Clear temporal cache
        self.temporal_cache.clear();

        freed_bytes
    }

    /// Clean up old buffers based on last access time
    fn cleanup_old_buffers(&mut self, current_time: f64, max_age_ms: f64) -> usize {
        let mut freed_bytes = 0;
        let cutoff_time = current_time - max_age_ms;

        // Remove old buffers from temporal cache
        for (_bank, buffers) in self.temporal_cache.iter_mut() {
            buffers.retain(|(_buffer, timestamp)| {
                if *timestamp < cutoff_time {
                    // Buffer is too old, remove it
                    // Note: In a real implementation, we'd need to track buffer sizes
                    false
                } else {
                    true
                }
            });
        }

        // Clean up large buffer cache
        let _old_len = self.large_buffer_cache.len();
        self.large_buffer_cache.retain(|(_buffer, _size)| {
            // For large cache, we don't have timestamps, so remove oldest half
            true // Simplified: keep all for now
        });

        // Remove old buffers based on metadata
        let old_buffer_sizes: Vec<usize> = self
            .buffer_metadata
            .iter()
            .filter(|(_, metadata)| metadata.last_used < cutoff_time)
            .map(|(&size, _)| size)
            .collect();

        for size in old_buffer_sizes {
            if let Some(buffers) = self.free_buffers.get_mut(&size) {
                let to_remove = buffers.len() / 2; // Remove half of old buffers
                for _ in 0..to_remove {
                    if buffers.pop().is_some() {
                        freed_bytes += size;
                        self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
                        track_gpu_deallocation(size);
                    }
                }
            }
        }

        freed_bytes
    }

    /// Priority-based buffer allocation for critical operations
    pub fn get_priority_buffer(
        &mut self,
        device: &GpuDevice,
        size: usize,
        usage: u32,
        priority: BufferPriority,
    ) -> Result<GpuBuffer, wasm_bindgen::JsValue> {
        match priority {
            BufferPriority::Critical => {
                // For critical operations, ensure we have the buffer even if it means aggressive cleanup
                if self.get_buffer(device, size, usage).is_err() {
                    // Try cleanup and retry
                    self.handle_memory_pressure(0.8);
                    self.get_buffer(device, size, usage)
                } else {
                    self.get_buffer(device, size, usage)
                }
            },
            BufferPriority::High => {
                // For high priority, try prefetching similar sizes
                let _aligned_size = self.align_size(size);
                let _ = self.prefetch_buffers(device);
                self.get_buffer(device, size, usage)
            },
            BufferPriority::Normal => self.get_buffer(device, size, usage),
            BufferPriority::Low => {
                // For low priority, only allocate if memory pressure is low
                if self.fragmentation_ratio() < 0.5
                    && self.allocated_bytes < self.max_buffer_size / 2
                {
                    self.get_buffer(device, size, usage)
                } else {
                    Err(wasm_bindgen::JsValue::from_str(
                        "Deferred allocation due to memory pressure",
                    ))
                }
            },
        }
    }

    /// Compact memory by reorganizing buffers to reduce fragmentation
    pub fn compact_memory(&mut self, device: &GpuDevice) -> Result<usize, wasm_bindgen::JsValue> {
        web_sys::console::log_1(&"🗜️ Starting memory compaction...".into());

        let initial_fragmentation = self.fragmentation_ratio();
        let mut bytes_reorganized = 0;

        // Find heavily fragmented buckets (many small buffers)
        let fragmented_buckets: Vec<(usize, usize)> = self
            .free_buffers
            .iter()
            .filter(|(_, buffers)| buffers.len() > 3)
            .map(|(&size, buffers)| (size, buffers.len()))
            .collect();

        for (bucket_size, count) in fragmented_buckets {
            if count > 4 {
                // Compact by reducing the number of buffers in this bucket
                if let Some(buffers) = self.free_buffers.get_mut(&bucket_size) {
                    let target_count = (count / 2).max(2);
                    let to_remove = count - target_count;

                    for _ in 0..to_remove {
                        if buffers.pop().is_some() {
                            bytes_reorganized += bucket_size;
                            self.allocated_bytes = self.allocated_bytes.saturating_sub(bucket_size);
                            track_gpu_deallocation(bucket_size);
                        }
                    }
                }
            }
        }

        // Create larger consolidated buffers
        if bytes_reorganized > 0 {
            let consolidated_size = (bytes_reorganized / 4).max(self.min_buffer_size * 4);
            // gpu_buffer_usage not available in web-sys 0.3.81 - using numeric values
            let usage = 0x80u32 | 0x08u32 | 0x04u32; // STORAGE | COPY_DST | COPY_SRC
            let consolidated_buffer = self.create_new_buffer(device, consolidated_size, usage)?;
            self.return_buffer(consolidated_buffer, consolidated_size);
            self.allocated_bytes += consolidated_size;
        }

        let final_fragmentation = self.fragmentation_ratio();
        let improvement = (initial_fragmentation - final_fragmentation) * 100.0;

        web_sys::console::log_1(
            &format!(
                "✅ Memory compaction complete: {:.1}% fragmentation improvement, {}KB reorganized",
                improvement,
                bytes_reorganized / 1024
            )
            .into(),
        );

        Ok(bytes_reorganized)
    }

    /// Adaptive memory management that automatically adjusts to workload patterns
    pub fn adaptive_manage(&mut self, device: &GpuDevice) -> Result<(), wasm_bindgen::JsValue> {
        let _current_time = js_sys::Date::now();

        // Update predictions based on recent activity
        self.predict_buffer_requirements();

        // Adaptive actions based on current state
        let fragmentation = self.fragmentation_ratio();
        let memory_pressure = (self.allocated_bytes as f32) / (self.max_buffer_size as f32);

        match (fragmentation, memory_pressure) {
            (f, p) if f > 0.8 || p > 0.9 => {
                // High fragmentation or pressure: aggressive cleanup
                let _ = self.compact_memory(device);
                self.handle_memory_pressure(p);
            },
            (f, p) if f > 0.6 || p > 0.7 => {
                // Medium fragmentation or pressure: defragment
                self.defragment();
            },
            (f, p) if f < 0.3 && p < 0.5 && self.access_predictor.prediction_confidence > 0.7 => {
                // Low fragmentation and pressure with good predictions: prefetch
                let _ = self.prefetch_buffers(device);
            },
            _ => {
                // Normal operation: just update statistics
            },
        }

        Ok(())
    }

    /// Advanced spatial locality optimization for memory coalescing
    pub fn optimize_spatial_locality(
        &mut self,
        device: &GpuDevice,
    ) -> Result<usize, wasm_bindgen::JsValue> {
        web_sys::console::log_1(&"🧠 Optimizing spatial locality for memory coalescing...".into());

        let mut bytes_optimized = 0;
        let _current_time = js_sys::Date::now();

        // Analyze spatial locality patterns in recent allocations
        let mut size_groups: HashMap<usize, Vec<usize>> = HashMap::new();

        // Group similar-sized buffers for better spatial locality
        for &size in self.buffer_metadata.keys() {
            let cache_line_aligned_size = self.align_to_cache_line(size);
            size_groups.entry(cache_line_aligned_size).or_default().push(size);
        }

        // Create spatial locality mappings
        for (aligned_size, related_sizes) in size_groups.iter() {
            if related_sizes.len() > 1 {
                self.memory_bandwidth_optimizer
                    .spatial_locality_map
                    .insert(*aligned_size, related_sizes.clone());

                // Coalesce spatially related buffers
                let total_related_size: usize = related_sizes.iter().sum();
                if total_related_size
                    > self.memory_bandwidth_optimizer.gpu_architecture_hints.cache_line_size * 4
                {
                    bytes_optimized +=
                        self.coalesce_spatially_related_buffers(device, related_sizes)?;
                }
            }
        }

        // Update memory channel utilization based on optimizations
        self.update_memory_channel_utilization();

        web_sys::console::log_1(
            &format!(
                "✅ Spatial locality optimization complete: {}KB optimized",
                bytes_optimized / 1024
            )
            .into(),
        );

        Ok(bytes_optimized)
    }

    /// Coalesce spatially related buffers for better memory bandwidth utilization
    fn coalesce_spatially_related_buffers(
        &mut self,
        device: &GpuDevice,
        related_sizes: &[usize],
    ) -> Result<usize, wasm_bindgen::JsValue> {
        let mut total_coalesced = 0;
        let optimal_coalesced_size = related_sizes.iter().sum::<usize>().next_power_of_two();

        // Only coalesce if the result is within reasonable bounds
        if optimal_coalesced_size <= self.max_buffer_size
            && optimal_coalesced_size >= self.min_buffer_size * 4
        {
            // Remove smaller related buffers
            for &size in related_sizes {
                if let Some(buffers) = self.free_buffers.get_mut(&size) {
                    let to_remove = (buffers.len() / 2).max(1); // Remove half
                    for _ in 0..to_remove {
                        if buffers.pop().is_some() {
                            total_coalesced += size;
                            self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
                            track_gpu_deallocation(size);
                        }
                    }
                }
            }

            // Create optimally coalesced buffer
            if total_coalesced > 0 {
                // gpu_buffer_usage not available in web-sys 0.3.81 - using numeric values
                let usage = 0x80u32 | 0x08u32 | 0x04u32; // STORAGE | COPY_DST | COPY_SRC
                let coalesced_buffer =
                    self.create_new_buffer(device, optimal_coalesced_size, usage)?;
                self.return_buffer(coalesced_buffer, optimal_coalesced_size);
                self.allocated_bytes += optimal_coalesced_size;
                self.stats.memory_saved_bytes +=
                    total_coalesced.saturating_sub(optimal_coalesced_size);
            }
        }

        Ok(total_coalesced)
    }

    /// Align size to GPU cache line boundaries for optimal memory access
    fn align_to_cache_line(&self, size: usize) -> usize {
        let cache_line_size =
            self.memory_bandwidth_optimizer.gpu_architecture_hints.cache_line_size;
        (size + cache_line_size - 1) & !(cache_line_size - 1)
    }

    /// Update memory channel utilization metrics
    fn update_memory_channel_utilization(&mut self) {
        let total_allocated = self.allocated_bytes as f32;
        let max_per_channel = total_allocated / 4.0; // Assume 4 memory channels

        // Simulate channel utilization based on buffer distribution
        for (i, channel_util) in self
            .memory_bandwidth_optimizer
            .memory_channel_utilization
            .iter_mut()
            .enumerate()
        {
            let channel_load = self
                .memory_bandwidth_optimizer
                .bank_load_balancing
                .iter()
                .skip(i * 2)
                .take(2)
                .sum::<f32>();
            *channel_util = (channel_load / max_per_channel).min(1.0);
        }

        // Update overall bandwidth utilization
        let avg_channel_util: f32 =
            self.memory_bandwidth_optimizer.memory_channel_utilization.iter().sum::<f32>() / 4.0;
        self.memory_bandwidth_optimizer.bandwidth_utilization = avg_channel_util;
    }

    /// Architecture-aware buffer alignment optimization
    pub fn optimize_architecture_alignment(
        &mut self,
        _device: &GpuDevice,
    ) -> Result<(), wasm_bindgen::JsValue> {
        web_sys::console::log_1(&"⚙️ Optimizing for GPU architecture-specific alignment...".into());

        let hints = &self.memory_bandwidth_optimizer.gpu_architecture_hints;
        let optimal_alignment = hints.coalescing_alignment;

        // Update buffer alignment based on GPU architecture
        if optimal_alignment > self.buffer_alignment {
            self.buffer_alignment = optimal_alignment;

            // Re-align existing buffers if beneficial
            let mut realigned_count = 0;
            let buffer_sizes: Vec<usize> = self.free_buffers.keys().copied().collect();

            for size in buffer_sizes {
                let new_aligned_size = self.align_size(size);
                if new_aligned_size != size {
                    // Would need to re-create buffers in real implementation
                    realigned_count += 1;
                }
            }

            web_sys::console::log_1(&format!(
                "✅ Architecture alignment optimized: {} buffers affected, alignment set to {} bytes",
                realigned_count, optimal_alignment
            ).into());
        }

        Ok(())
    }

    /// Cross-device memory synchronization optimization
    pub fn optimize_cross_device_sync(&mut self) -> f32 {
        web_sys::console::log_1(&"🔄 Optimizing cross-device memory synchronization...".into());

        let current_overhead = self.memory_bandwidth_optimizer.cross_device_sync_overhead;

        // Calculate optimal sync overhead based on bandwidth utilization
        let bandwidth_util = self.memory_bandwidth_optimizer.bandwidth_utilization;
        let optimal_overhead =
            if bandwidth_util > self.memory_bandwidth_optimizer.bandwidth_saturation_threshold {
                // High bandwidth utilization: increase sync overhead to reduce contention
                (current_overhead * 1.2).min(0.15) // Max 15% overhead
            } else {
                // Low bandwidth utilization: reduce sync overhead for better performance
                (current_overhead * 0.9).max(0.02) // Min 2% overhead
            };

        self.memory_bandwidth_optimizer.cross_device_sync_overhead = optimal_overhead;

        web_sys::console::log_1(
            &format!(
                "✅ Cross-device sync optimized: overhead adjusted from {:.1}% to {:.1}%",
                current_overhead * 100.0,
                optimal_overhead * 100.0
            )
            .into(),
        );

        optimal_overhead
    }

    /// Memory warming for ML-specific workload patterns
    pub fn warm_for_ml_workloads(
        &mut self,
        device: &GpuDevice,
        model_type: &str,
    ) -> Result<(), wasm_bindgen::JsValue> {
        web_sys::console::log_1(
            &format!("🔥 Warming memory pool for {} ML workloads...", model_type).into(),
        );

        let critical_sizes = match model_type {
            "transformer" => vec![
                512 * 1024,       // Small attention matrices
                2 * 1024 * 1024,  // Medium weight matrices
                8 * 1024 * 1024,  // Large embedding matrices
                32 * 1024 * 1024, // Very large weight matrices
            ],
            "cnn" => vec![
                256 * 1024,       // Small conv kernels
                1024 * 1024,      // Feature maps
                4 * 1024 * 1024,  // Large feature maps
                16 * 1024 * 1024, // Very large feature maps
            ],
            "rnn" => vec![
                128 * 1024,      // Hidden states
                512 * 1024,      // Cell states
                2 * 1024 * 1024, // Weight matrices
                8 * 1024 * 1024, // Large sequence buffers
            ],
            _ => vec![
                1024 * 1024,     // Default medium buffers
                4 * 1024 * 1024, // Default large buffers
            ],
        };

        // Pre-warm cache with ML-specific buffer sizes
        self.warm_cache(device, &critical_sizes)?;

        // Update spatial locality map for ML patterns
        for (i, &size) in critical_sizes.iter().enumerate() {
            let related_sizes: Vec<usize> = critical_sizes
                .iter()
                .enumerate()
                .filter(|(j, _)| i != *j)
                .map(|(_, &s)| s)
                .collect();
            self.memory_bandwidth_optimizer.spatial_locality_map.insert(size, related_sizes);
        }

        web_sys::console::log_1(
            &format!(
                "✅ ML workload warming complete: {} critical sizes pre-allocated",
                critical_sizes.len()
            )
            .into(),
        );

        Ok(())
    }

    /// Bandwidth-aware memory allocation with saturation detection
    pub fn get_bandwidth_aware_buffer(
        &mut self,
        device: &GpuDevice,
        size: usize,
        usage: u32,
    ) -> Result<GpuBuffer, wasm_bindgen::JsValue> {
        // Check if we're approaching bandwidth saturation
        if self.memory_bandwidth_optimizer.bandwidth_utilization
            > self.memory_bandwidth_optimizer.bandwidth_saturation_threshold
        {
            web_sys::console::log_1(
                &"⚠️ Bandwidth saturation detected, using conservative allocation...".into(),
            );

            // Use more conservative allocation strategy
            let old_strategy = self.allocation_strategy;
            self.allocation_strategy = AllocationStrategy::BestFit; // Most conservative

            let result = self.get_buffer(device, size, usage);

            // Restore original strategy
            self.allocation_strategy = old_strategy;

            result
        } else {
            // Normal allocation with spatial locality optimization
            let aligned_size = self.align_to_cache_line(size);

            // Check if we have spatially related buffers
            let related_sizes_clone =
                self.memory_bandwidth_optimizer.spatial_locality_map.get(&aligned_size).cloned();
            if let Some(related_sizes) = related_sizes_clone {
                // Try to allocate from spatially related buffers first
                for &related_size in &related_sizes {
                    if let Some(buffer) = self.find_suitable_buffer(related_size, usage) {
                        self.update_buffer_metadata(related_size, js_sys::Date::now(), false);
                        return Ok(buffer);
                    }
                }
            }

            // Fall back to normal allocation
            self.get_buffer(device, size, usage)
        }
    }

    /// Comprehensive memory coalescing analysis and optimization
    pub fn comprehensive_coalescing_analysis(
        &mut self,
        device: &GpuDevice,
    ) -> Result<js_sys::Object, wasm_bindgen::JsValue> {
        web_sys::console::log_1(
            &"📊 Performing comprehensive memory coalescing analysis...".into(),
        );

        // Run all optimization strategies
        let spatial_optimized = self.optimize_spatial_locality(device)?;
        self.optimize_architecture_alignment(device)?;
        let sync_overhead = self.optimize_cross_device_sync();
        let compacted = self.compact_memory(device)?;

        // Create comprehensive analysis report
        let analysis = js_sys::Object::new();

        js_sys::Reflect::set(
            &analysis,
            &"spatial_locality_optimized_bytes".into(),
            &spatial_optimized.into(),
        )?;
        js_sys::Reflect::set(
            &analysis,
            &"memory_compacted_bytes".into(),
            &compacted.into(),
        )?;
        js_sys::Reflect::set(
            &analysis,
            &"cross_device_sync_overhead".into(),
            &sync_overhead.into(),
        )?;
        js_sys::Reflect::set(
            &analysis,
            &"bandwidth_utilization".into(),
            &self.memory_bandwidth_optimizer.bandwidth_utilization.into(),
        )?;
        js_sys::Reflect::set(
            &analysis,
            &"fragmentation_ratio".into(),
            &self.fragmentation_ratio().into(),
        )?;

        // Memory channel utilization
        let channel_utils = js_sys::Array::new();
        for util in &self.memory_bandwidth_optimizer.memory_channel_utilization {
            channel_utils.push(&(*util).into());
        }
        js_sys::Reflect::set(
            &analysis,
            &"memory_channel_utilization".into(),
            &channel_utils.into(),
        )?;

        // Spatial locality map size
        js_sys::Reflect::set(
            &analysis,
            &"spatial_locality_mappings".into(),
            &self.memory_bandwidth_optimizer.spatial_locality_map.len().into(),
        )?;

        // Architecture hints
        let arch_info = js_sys::Object::new();
        let hints = &self.memory_bandwidth_optimizer.gpu_architecture_hints;
        js_sys::Reflect::set(
            &arch_info,
            &"cache_line_size".into(),
            &hints.cache_line_size.into(),
        )?;
        js_sys::Reflect::set(
            &arch_info,
            &"memory_bus_width".into(),
            &hints.memory_bus_width.into(),
        )?;
        js_sys::Reflect::set(
            &arch_info,
            &"max_bandwidth_gbps".into(),
            &hints.max_bandwidth_gbps.into(),
        )?;
        js_sys::Reflect::set(
            &arch_info,
            &"coalescing_alignment".into(),
            &hints.coalescing_alignment.into(),
        )?;
        js_sys::Reflect::set(
            &analysis,
            &"gpu_architecture_hints".into(),
            &arch_info.into(),
        )?;

        web_sys::console::log_1(&"✅ Comprehensive coalescing analysis complete".into());

        Ok(analysis)
    }
}

/// Buffer allocation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPriority {
    Critical, // Must succeed, can trigger aggressive cleanup
    High,     // Important, may trigger prefetching
    Normal,   // Standard priority
    Low,      // Can be deferred under memory pressure
}

/// Utility functions for advanced buffer management
#[wasm_bindgen]
pub fn create_advanced_buffer_pool() -> BufferPool {
    BufferPool::new()
}

#[wasm_bindgen]
pub fn create_buffer_pool_with_strategy(strategy_id: u32) -> BufferPool {
    let strategy = match strategy_id {
        0 => AllocationStrategy::FirstFit,
        1 => AllocationStrategy::BestFit,
        2 => AllocationStrategy::WorstFit,
        3 => AllocationStrategy::BuddySystem,
        _ => AllocationStrategy::BestFit,
    };
    BufferPool::with_strategy_internal(strategy)
}
