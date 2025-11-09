//! Bandwidth optimization and traffic management for mobile federated learning.
//!
//! This module provides comprehensive bandwidth optimization capabilities including
//! intelligent compression algorithms, traffic shaping, rate limiting, and data usage
//! tracking for efficient mobile federated learning operations.

use std::collections::{HashMap, VecDeque};
use trustformers_core::Result;

use super::types::{
    AllocationStrategy, CompressionStats, DeltaAlgorithm, DeltaCompressionConfig,
    GradientCompressionAlgorithm, NetworkAdaptationConfig, NetworkQuantizationConfig,
    PruningConfig,
};

/// Bandwidth optimizer for federated learning communications
pub struct BandwidthOptimizer {
    config: NetworkAdaptationConfig,
    compression_engine: NetworkCompressionEngine,
    traffic_shaper: TrafficShaper,
    usage_tracker: DataUsageTracker,
}

/// Error feedback buffer for tracking compression errors
#[derive(Debug, Clone, Default)]
pub struct ErrorFeedbackBuffer {
    pub error_count: u32,
    pub last_error_timestamp: Option<std::time::Instant>,
    pub error_rate: f32,
}

/// Network compression engine for model and gradient optimization
pub struct NetworkCompressionEngine {
    gradient_compressor: GradientCompressor,
    model_compressor: ModelCompressor,
    differential_compressor: DifferentialCompressor,
    compression_stats: CompressionStats,
}

/// Gradient compression for federated learning
pub struct GradientCompressor {
    algorithm: GradientCompressionAlgorithm,
    error_feedback: ErrorFeedbackBuffer,
    compression_ratio: f32,
    quality_threshold: f32,
}

/// Model compression with quantization and pruning
pub struct ModelCompressor {
    quantization_config: NetworkQuantizationConfig,
    pruning_config: PruningConfig,
    delta_compression: DeltaCompressionConfig,
}

/// Differential compression for model updates
pub struct DifferentialCompressor {
    baseline_models: HashMap<String, Vec<u8>>,
    delta_algorithm: DeltaAlgorithm,
    compression_cache: CompressionCache,
}

/// Compression cache for performance optimization
pub struct CompressionCache {
    cached_deltas: HashMap<String, Vec<u8>>,
    cache_hit_rate: f32,
    max_cache_size_mb: u32,
}

/// Traffic shaping and bandwidth allocation
pub struct TrafficShaper {
    rate_limiter: RateLimiter,
    priority_queues: HashMap<String, VecDeque<Vec<u8>>>,
    bandwidth_allocator: BandwidthAllocator,
}

/// Rate limiting for network traffic
pub struct RateLimiter {
    current_rate_mbps: f32,
    target_rate_mbps: f32,
    burst_allowance_mb: f32,
    window_size_ms: u64,
}

/// Bandwidth allocation based on task priority
pub struct BandwidthAllocator {
    total_bandwidth_mbps: f32,
    allocated_bandwidth: HashMap<String, f32>,
    allocation_strategy: AllocationStrategy,
}

/// Data usage tracking and prediction
pub struct DataUsageTracker {
    daily_usage: HashMap<String, u64>,
    monthly_usage: HashMap<String, u64>,
    usage_history: VecDeque<(std::time::Instant, u64)>,
    usage_predictor: UsagePredictor,
}

/// Usage prediction for proactive optimization
pub struct UsagePredictor {
    prediction_models: HashMap<String, Vec<f32>>,
    prediction_accuracy: f32,
    prediction_window_hours: u32,
}

impl BandwidthOptimizer {
    /// Create new bandwidth optimizer
    pub fn new(config: NetworkAdaptationConfig) -> Result<Self> {
        Ok(Self {
            config,
            compression_engine: NetworkCompressionEngine::new(),
            traffic_shaper: TrafficShaper::new(),
            usage_tracker: DataUsageTracker::new(),
        })
    }

    /// Start bandwidth optimization
    pub fn start(&mut self) -> Result<()> {
        // Initialize optimization subsystems
        // In a real implementation, this would start background optimization threads
        Ok(())
    }

    /// Stop bandwidth optimization
    pub fn stop(&mut self) -> Result<()> {
        // Stop optimization subsystems
        Ok(())
    }

    /// Optimize data for transmission
    pub fn optimize_transmission(&mut self, data: Vec<u8>, data_type: &str) -> Result<Vec<u8>> {
        // Apply compression based on data type
        let compressed_data = match data_type {
            "gradient" => self.compression_engine.compress_gradient(data)?,
            "model" => self.compression_engine.compress_model(data)?,
            "differential" => self.compression_engine.compress_differential(data)?,
            _ => data, // No compression for unknown types
        };

        // Apply traffic shaping
        self.traffic_shaper.shape_traffic(compressed_data, data_type)
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> &CompressionStats {
        &self.compression_engine.compression_stats
    }

    /// Get current bandwidth utilization
    pub fn get_bandwidth_utilization(&self) -> f32 {
        self.traffic_shaper.get_current_utilization()
    }

    /// Get data usage statistics
    pub fn get_usage_stats(&self) -> HashMap<String, u64> {
        self.usage_tracker.get_current_usage()
    }

    /// Predict future bandwidth requirements
    pub fn predict_bandwidth_requirements(&self, hours: u32) -> Result<f32> {
        self.usage_tracker.predict_usage(hours)
    }

    /// Update optimization configuration
    pub fn update_config(&mut self, config: NetworkAdaptationConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Check if bandwidth limit is approaching
    pub fn is_approaching_limit(&self) -> bool {
        self.usage_tracker.is_approaching_limit(&self.config.data_usage_limits)
    }

    /// Get recommended compression level
    pub fn get_recommended_compression_level(&self) -> f32 {
        // Calculate compression level based on current conditions
        let usage_ratio = self.usage_tracker.get_usage_ratio(&self.config.data_usage_limits);
        let bandwidth_ratio = self.traffic_shaper.get_utilization_ratio();

        // Higher usage/utilization -> higher compression
        (usage_ratio + bandwidth_ratio) / 2.0
    }
}

impl NetworkCompressionEngine {
    /// Create new compression engine
    pub fn new() -> Self {
        Self {
            gradient_compressor: GradientCompressor::new(),
            model_compressor: ModelCompressor::new(),
            differential_compressor: DifferentialCompressor::new(),
            compression_stats: CompressionStats::default(),
        }
    }

    /// Compress gradient data
    pub fn compress_gradient(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        let original_size = data.len();
        let compressed = self.gradient_compressor.compress(data)?;
        self.update_stats(original_size, compressed.len());
        Ok(compressed)
    }

    /// Compress model data
    pub fn compress_model(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        let original_size = data.len();
        let compressed = self.model_compressor.compress(data)?;
        self.update_stats(original_size, compressed.len());
        Ok(compressed)
    }

    /// Compress differential data
    pub fn compress_differential(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        let original_size = data.len();
        let compressed = self.differential_compressor.compress(data)?;
        self.update_stats(original_size, compressed.len());
        Ok(compressed)
    }

    /// Update compression statistics
    fn update_stats(&mut self, original_size: usize, compressed_size: usize) {
        self.compression_stats.original_size_bytes += original_size;
        self.compression_stats.compressed_size_bytes += compressed_size;

        if self.compression_stats.original_size_bytes > 0 {
            self.compression_stats.compression_ratio = self.compression_stats.compressed_size_bytes
                as f32
                / self.compression_stats.original_size_bytes as f32;
        }
    }

    /// Get compression efficiency
    pub fn get_compression_efficiency(&self) -> f32 {
        1.0 - self.compression_stats.compression_ratio
    }
}

impl GradientCompressor {
    /// Create new gradient compressor
    pub fn new() -> Self {
        Self {
            algorithm: GradientCompressionAlgorithm::Adaptive,
            error_feedback: ErrorFeedbackBuffer::default(),
            compression_ratio: 0.7,
            quality_threshold: 0.95,
        }
    }

    /// Compress gradient data
    pub fn compress(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Apply compression algorithm based on configuration
        match self.algorithm {
            GradientCompressionAlgorithm::TopK { .. } => self.compress_top_k(data),
            GradientCompressionAlgorithm::RandomSparsification { .. } => {
                self.compress_randomized(data)
            },
            GradientCompressionAlgorithm::Adaptive => self.compress_adaptive(data),
            GradientCompressionAlgorithm::Quantized { .. } => self.compress_quantized(data),
            GradientCompressionAlgorithm::None => Ok(data), // No compression
            GradientCompressionAlgorithm::ThresholdBased { .. } => self.compress_top_k(data),
        }
    }

    /// Apply top-k compression
    fn compress_top_k(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Simplified top-k compression
        let keep_ratio = self.compression_ratio;
        let keep_count = (data.len() as f32 * keep_ratio) as usize;
        Ok(data.into_iter().take(keep_count).collect())
    }

    /// Apply randomized compression
    fn compress_randomized(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Simplified randomized compression
        let keep_ratio = self.compression_ratio;
        let data_len = data.len();
        let keep_count = (data_len as f32 * keep_ratio) as usize;
        let step_size = if keep_count > 0 { data_len / keep_count } else { 1 };
        Ok(data.into_iter().step_by(step_size).collect())
    }

    /// Apply adaptive compression
    fn compress_adaptive(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Use error feedback to adapt compression
        let feedback_adjustment = 1.0; // Simple placeholder for error feedback
        let adjusted_ratio = (self.compression_ratio + feedback_adjustment).clamp(0.1, 0.9);

        let keep_count = (data.len() as f32 * adjusted_ratio) as usize;
        Ok(data.into_iter().take(keep_count).collect())
    }

    /// Apply quantization compression
    fn compress_quantized(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Simplified quantization (reduce bit precision)
        Ok(data.into_iter().map(|b| (b / 4) * 4).collect())
    }

    /// Update compression parameters
    pub fn update_parameters(&mut self, compression_ratio: f32, quality_threshold: f32) {
        self.compression_ratio = compression_ratio.clamp(0.1, 0.9);
        self.quality_threshold = quality_threshold.clamp(0.5, 1.0);
    }
}

impl ModelCompressor {
    /// Create new model compressor
    pub fn new() -> Self {
        Self {
            quantization_config: NetworkQuantizationConfig::default(),
            pruning_config: PruningConfig::default(),
            delta_compression: DeltaCompressionConfig::default(),
        }
    }

    /// Compress model data
    pub fn compress(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        let mut compressed_data = data;

        // Apply pruning if enabled
        if self.pruning_config.enable_pruning {
            compressed_data = self.apply_pruning(compressed_data)?;
        }

        // Apply quantization
        compressed_data = self.apply_quantization(compressed_data)?;

        // Apply delta compression if enabled
        if self.delta_compression.enable_delta {
            compressed_data = self.apply_delta_compression(compressed_data)?;
        }

        Ok(compressed_data)
    }

    /// Apply pruning to reduce model size
    fn apply_pruning(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        let pruning_ratio = self.pruning_config.pruning_ratio;
        let keep_count = (data.len() as f32 * (1.0 - pruning_ratio)) as usize;

        if self.pruning_config.structured_pruning {
            // Structured pruning - remove entire blocks
            let block_size = data.len() / keep_count;
            Ok(data.chunks(block_size).take(keep_count).flatten().copied().collect())
        } else {
            // Unstructured pruning - remove individual elements
            Ok(data.into_iter().step_by((1.0 / (1.0 - pruning_ratio)) as usize).collect())
        }
    }

    /// Apply quantization to reduce precision
    fn apply_quantization(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Simplified quantization - reduce bit precision
        let bits = self.quantization_config.gradient_bits;
        let quantization_factor = 256 / (1 << bits);

        Ok(data
            .into_iter()
            .map(|b| (b / quantization_factor as u8) * quantization_factor as u8)
            .collect())
    }

    /// Apply delta compression
    fn apply_delta_compression(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Simplified delta compression - could be much more sophisticated
        match self.delta_compression.delta_algorithm {
            DeltaAlgorithm::SimpleDiff => {
                // Apply simple differencing
                let mut result = Vec::with_capacity(data.len());
                if !data.is_empty() {
                    result.push(data[0]);
                    for i in 1..data.len() {
                        result.push(data[i].wrapping_sub(data[i - 1]));
                    }
                }
                Ok(result)
            },
            DeltaAlgorithm::OptimizedDiff => {
                // More sophisticated delta compression would go here
                Ok(data)
            },
            DeltaAlgorithm::SemanticDiff => {
                // Semantic diff - fallback to simple diff for now
                let mut result = Vec::with_capacity(data.len());
                if !data.is_empty() {
                    result.push(data[0]);
                    for i in 1..data.len() {
                        result.push(data[i].wrapping_sub(data[i - 1]));
                    }
                }
                Ok(result)
            },
            DeltaAlgorithm::CompressedDiff => {
                // Compressed diff - fallback to optimized diff for now
                Ok(data)
            },
        }
    }
}

impl DifferentialCompressor {
    /// Create new differential compressor
    pub fn new() -> Self {
        Self {
            baseline_models: HashMap::new(),
            delta_algorithm: DeltaAlgorithm::OptimizedDiff,
            compression_cache: CompressionCache::new(),
        }
    }

    /// Compress data using differential method
    pub fn compress(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Check cache first
        let data_hash = self.calculate_hash(&data);
        if let Some(cached) = self.compression_cache.get(&data_hash) {
            return Ok(cached);
        }

        // Apply differential compression
        let compressed = match self.delta_algorithm {
            DeltaAlgorithm::SimpleDiff => self.simple_diff_compress(data)?,
            DeltaAlgorithm::OptimizedDiff => self.optimized_diff_compress(data)?,
            DeltaAlgorithm::SemanticDiff => self.simple_diff_compress(data)?, // Fallback to simple diff
            DeltaAlgorithm::CompressedDiff => self.optimized_diff_compress(data)?, // Fallback to optimized diff
        };

        // Cache the result
        self.compression_cache.insert(data_hash, compressed.clone());

        Ok(compressed)
    }

    /// Simple differential compression
    fn simple_diff_compress(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Find the most similar baseline model
        if let Some((_, baseline)) = self.baseline_models.iter().next() {
            let mut diff = Vec::new();
            for (i, &byte) in data.iter().enumerate() {
                if i < baseline.len() {
                    diff.push(byte.wrapping_sub(baseline[i]));
                } else {
                    diff.push(byte);
                }
            }
            Ok(diff)
        } else {
            Ok(data)
        }
    }

    /// Optimized differential compression
    fn optimized_diff_compress(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // More sophisticated differential compression algorithm
        // For now, just return simple diff
        self.simple_diff_compress(data)
    }

    /// Calculate simple hash for caching
    fn calculate_hash(&self, data: &[u8]) -> String {
        // Simplified hash - in practice, use a proper hash function
        format!(
            "{:08x}",
            (data.len() as u32) ^ data.iter().fold(0u32, |acc, &b| acc.wrapping_add(b as u32))
        )
    }

    /// Set baseline model for differential compression
    pub fn set_baseline(&mut self, model_id: String, baseline: Vec<u8>) {
        self.baseline_models.insert(model_id, baseline);
    }
}

impl Default for CompressionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionCache {
    /// Create new compression cache
    pub fn new() -> Self {
        Self {
            cached_deltas: HashMap::new(),
            cache_hit_rate: 0.0,
            max_cache_size_mb: 100,
        }
    }

    /// Get cached compression result
    pub fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        self.cached_deltas.get(key).cloned()
    }

    /// Insert compression result into cache
    pub fn insert(&mut self, key: String, value: Vec<u8>) {
        // Simple cache management - remove oldest if at capacity
        let current_size_mb = self.get_cache_size_mb();
        if current_size_mb >= self.max_cache_size_mb {
            self.evict_oldest();
        }

        self.cached_deltas.insert(key, value);
    }

    /// Get current cache size in MB
    fn get_cache_size_mb(&self) -> u32 {
        let total_bytes: usize = self.cached_deltas.values().map(|v| v.len()).sum();
        (total_bytes / (1024 * 1024)) as u32
    }

    /// Evict oldest cache entry
    fn evict_oldest(&mut self) {
        // Simplified eviction - just remove first entry
        if let Some(key) = self.cached_deltas.keys().next().cloned() {
            self.cached_deltas.remove(&key);
        }
    }
}

impl TrafficShaper {
    /// Create new traffic shaper
    pub fn new() -> Self {
        Self {
            rate_limiter: RateLimiter::new(),
            priority_queues: HashMap::new(),
            bandwidth_allocator: BandwidthAllocator::new(),
        }
    }

    /// Shape traffic based on priority and bandwidth allocation
    pub fn shape_traffic(&mut self, data: Vec<u8>, data_type: &str) -> Result<Vec<u8>> {
        // Check rate limits
        if !self.rate_limiter.check_rate_limit(data.len()) {
            // Queue data if rate limit exceeded
            self.queue_data(data_type.to_string(), data.clone());
            return Ok(Vec::new()); // Return empty to indicate queuing
        }

        // Apply bandwidth allocation
        self.bandwidth_allocator.allocate_bandwidth(data_type, data.len() as f32);

        Ok(data)
    }

    /// Queue data for later transmission
    fn queue_data(&mut self, data_type: String, data: Vec<u8>) {
        self.priority_queues.entry(data_type).or_default().push_back(data);
    }

    /// Get current bandwidth utilization
    pub fn get_current_utilization(&self) -> f32 {
        self.rate_limiter.current_rate_mbps / self.rate_limiter.target_rate_mbps
    }

    /// Get utilization ratio (0.0 to 1.0)
    pub fn get_utilization_ratio(&self) -> f32 {
        self.get_current_utilization().clamp(0.0, 1.0)
    }

    /// Process queued data
    pub fn process_queue(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut processed = Vec::new();

        for (_, queue) in self.priority_queues.iter_mut() {
            while let Some(data) = queue.pop_front() {
                if self.rate_limiter.check_rate_limit(data.len()) {
                    processed.push(data);
                } else {
                    queue.push_front(data); // Put back if rate limited
                    break;
                }
            }
        }

        Ok(processed)
    }
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new() -> Self {
        Self {
            current_rate_mbps: 0.0,
            target_rate_mbps: 10.0,
            burst_allowance_mb: 5.0,
            window_size_ms: 1000,
        }
    }

    /// Check if data transmission is within rate limits
    pub fn check_rate_limit(&mut self, data_size_bytes: usize) -> bool {
        let data_size_mb = data_size_bytes as f32 / (1024.0 * 1024.0);

        // Simple rate limiting - in practice, this would be more sophisticated
        if self.current_rate_mbps + data_size_mb <= self.target_rate_mbps + self.burst_allowance_mb
        {
            self.current_rate_mbps += data_size_mb;
            true
        } else {
            false
        }
    }

    /// Update rate limits
    pub fn update_target_rate(&mut self, target_mbps: f32) {
        self.target_rate_mbps = target_mbps.max(0.1); // Minimum 0.1 Mbps
    }

    /// Reset rate counters (typically called periodically)
    pub fn reset_counters(&mut self) {
        self.current_rate_mbps = 0.0;
    }
}

impl BandwidthAllocator {
    /// Create new bandwidth allocator
    pub fn new() -> Self {
        Self {
            total_bandwidth_mbps: 10.0,
            allocated_bandwidth: HashMap::new(),
            allocation_strategy: AllocationStrategy::PriorityBased,
        }
    }

    /// Allocate bandwidth for specific data type
    pub fn allocate_bandwidth(&mut self, data_type: &str, data_size_mb: f32) {
        let current = self.allocated_bandwidth.entry(data_type.to_string()).or_insert(0.0);
        *current += data_size_mb;
    }

    /// Get available bandwidth
    pub fn get_available_bandwidth(&self) -> f32 {
        let allocated: f32 = self.allocated_bandwidth.values().sum();
        (self.total_bandwidth_mbps - allocated).max(0.0)
    }

    /// Update total bandwidth
    pub fn update_total_bandwidth(&mut self, bandwidth_mbps: f32) {
        self.total_bandwidth_mbps = bandwidth_mbps.max(0.1);
    }

    /// Reset allocations (typically called periodically)
    pub fn reset_allocations(&mut self) {
        self.allocated_bandwidth.clear();
    }
}

impl DataUsageTracker {
    /// Create new data usage tracker
    pub fn new() -> Self {
        Self {
            daily_usage: HashMap::new(),
            monthly_usage: HashMap::new(),
            usage_history: VecDeque::new(),
            usage_predictor: UsagePredictor::new(),
        }
    }

    /// Track data usage
    pub fn track_usage(&mut self, data_type: &str, bytes: u64) {
        // Update daily usage
        let daily_entry = self.daily_usage.entry(data_type.to_string()).or_insert(0);
        *daily_entry += bytes;

        // Update monthly usage
        let monthly_entry = self.monthly_usage.entry(data_type.to_string()).or_insert(0);
        *monthly_entry += bytes;

        // Add to history
        self.usage_history.push_back((std::time::Instant::now(), bytes));
        if self.usage_history.len() > 1000 {
            self.usage_history.pop_front();
        }
    }

    /// Get current usage statistics
    pub fn get_current_usage(&self) -> HashMap<String, u64> {
        self.daily_usage.clone()
    }

    /// Predict future usage
    pub fn predict_usage(&self, hours: u32) -> Result<f32> {
        self.usage_predictor.predict(hours, &self.usage_history)
    }

    /// Check if approaching data limits
    pub fn is_approaching_limit(&self, limits: &super::types::DataUsageLimits) -> bool {
        let total_daily: u64 = self.daily_usage.values().sum();
        let total_monthly: u64 = self.monthly_usage.values().sum();

        total_daily > (limits.cellular_daily_limit_mb.unwrap_or(0) * 1024 * 1024) as u64 * 80 / 100
            || total_monthly
                > (limits.cellular_monthly_limit_mb.unwrap_or(0) * 1024 * 1024) as u64 * 80 / 100
    }

    /// Get usage ratio compared to limits
    pub fn get_usage_ratio(&self, limits: &super::types::DataUsageLimits) -> f32 {
        let total_daily: u64 = self.daily_usage.values().sum();
        let daily_ratio = total_daily as f32
            / ((limits.cellular_daily_limit_mb.unwrap_or(0) * 1024 * 1024) as f32);
        daily_ratio.clamp(0.0, 1.0)
    }
}

impl UsagePredictor {
    /// Create new usage predictor
    pub fn new() -> Self {
        Self {
            prediction_models: HashMap::new(),
            prediction_accuracy: 0.5,
            prediction_window_hours: 24,
        }
    }

    /// Predict usage for the next N hours
    pub fn predict(
        &self,
        hours: u32,
        history: &VecDeque<(std::time::Instant, u64)>,
    ) -> Result<f32> {
        if history.is_empty() {
            return Ok(0.0);
        }

        // Simple prediction based on recent average
        let recent_count = (history.len() / 4).max(1); // Use last 25% of data
        let recent_usage: u64 =
            history.iter().rev().take(recent_count).map(|(_, usage)| usage).sum();

        let average_usage_per_hour = recent_usage as f32 / recent_count as f32;
        Ok(average_usage_per_hour * hours as f32)
    }

    /// Update prediction accuracy
    pub fn update_accuracy(&mut self, predicted: f32, actual: f32) {
        let error = (predicted - actual).abs() / actual.max(1.0);
        let accuracy = 1.0 - error;

        // Update running average of accuracy
        self.prediction_accuracy = (self.prediction_accuracy * 0.9) + (accuracy * 0.1);
        self.prediction_accuracy = self.prediction_accuracy.clamp(0.0, 1.0);
    }
}

// Default implementations for convenience
impl Default for NetworkCompressionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for GradientCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ModelCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DifferentialCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TrafficShaper {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BandwidthAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DataUsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for UsagePredictor {
    fn default() -> Self {
        Self::new()
    }
}
