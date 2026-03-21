//! Adaptive Inference for Mobile Deployment
//!
//! This module implements adaptive inference strategies that dynamically adjust
//! model computation based on device capabilities, power status, thermal conditions,
//! and performance requirements to provide optimal user experience.

use crate::{MobileBackend, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use trustformers_core::errors::runtime_error;
use trustformers_core::Tensor;

/// Adaptive inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable dynamic quality adjustment
    pub dynamic_quality: bool,
    /// Enable early exit strategies
    pub early_exit: bool,
    /// Enable progressive inference
    pub progressive_inference: bool,
    /// Enable cascade inference
    pub cascade_inference: bool,
    /// Minimum quality threshold (0.0 - 1.0)
    pub min_quality_threshold: f32,
    /// Maximum latency tolerance (ms)
    pub max_latency_ms: u64,
    /// Power-aware adjustments
    pub power_aware: bool,
    /// Thermal-aware adjustments
    pub thermal_aware: bool,
    /// Network-aware adjustments
    pub network_aware: bool,
    /// Cache inference results
    pub enable_caching: bool,
    /// Batch similar requests
    pub enable_batching: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            dynamic_quality: true,
            early_exit: true,
            progressive_inference: false,
            cascade_inference: false,
            min_quality_threshold: 0.7,
            max_latency_ms: 100,
            power_aware: true,
            thermal_aware: true,
            network_aware: false,
            enable_caching: true,
            enable_batching: true,
        }
    }
}

/// Adaptive inference strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceStrategy {
    /// Full model inference
    Full,
    /// Early exit at intermediate layers
    EarlyExit,
    /// Progressive inference with increasing complexity
    Progressive,
    /// Cascade of models from simple to complex
    Cascade,
    /// Dynamic quality adjustment
    DynamicQuality,
    /// Cached result reuse
    Cached,
}

/// Device capability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// CPU performance score (0.0 - 1.0)
    pub cpu_performance: f32,
    /// GPU performance score (0.0 - 1.0)
    pub gpu_performance: f32,
    /// Available memory (MB)
    pub available_memory_mb: u32,
    /// Battery level (0.0 - 1.0)
    pub battery_level: f32,
    /// Thermal state
    pub thermal_state: ThermalState,
    /// Power source
    pub power_source: PowerSource,
    /// Network connectivity
    pub network_state: NetworkState,
}

/// Thermal states
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ThermalState {
    Normal,
    Warm,
    Hot,
    Critical,
}

/// Power sources
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PowerSource {
    Battery,
    Charging,
    Plugged,
}

/// Network states
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NetworkState {
    Offline,
    Cellular,
    WiFi,
    Ethernet,
}

/// Adaptive inference engine
pub struct AdaptiveInferenceEngine {
    config: AdaptiveConfig,
    backend: MobileBackend,
    device_monitor: Arc<Mutex<DeviceMonitor>>,
    performance_predictor: PerformancePredictor,
    quality_controller: QualityController,
    cache_manager: CacheManager,
    batch_scheduler: BatchScheduler,
    stats: AdaptiveStats,
}

impl AdaptiveInferenceEngine {
    /// Create new adaptive inference engine
    pub fn new(config: AdaptiveConfig, backend: MobileBackend) -> Result<Self> {
        let device_monitor = Arc::new(Mutex::new(DeviceMonitor::new()?));
        let performance_predictor = PerformancePredictor::new();
        let quality_controller = QualityController::new(config.min_quality_threshold);
        let cache_manager = CacheManager::new(config.enable_caching);
        let batch_scheduler = BatchScheduler::new(config.enable_batching);

        Ok(Self {
            config,
            backend,
            device_monitor,
            performance_predictor,
            quality_controller,
            cache_manager,
            batch_scheduler,
            stats: AdaptiveStats::default(),
        })
    }

    /// Perform adaptive inference
    pub fn infer(&mut self, input: &Tensor, context: InferenceContext) -> Result<InferenceResult> {
        let start_time = Instant::now();

        // Get current device capabilities
        let capabilities = self
            .device_monitor
            .lock()
            .expect("device_monitor lock should not be poisoned")
            .get_capabilities()?;

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_result) = self.cache_manager.get(input, &context)? {
                self.stats.cache_hits += 1;
                return Ok(cached_result);
            }
            self.stats.cache_misses += 1;
        }

        // Determine optimal inference strategy
        let strategy = self.select_strategy(&capabilities, &context)?;

        // Perform inference with selected strategy
        let result = match strategy {
            InferenceStrategy::Full => self.full_inference(input, &context)?,
            InferenceStrategy::EarlyExit => self.early_exit_inference(input, &context)?,
            InferenceStrategy::Progressive => self.progressive_inference(input, &context)?,
            InferenceStrategy::Cascade => self.cascade_inference(input, &context)?,
            InferenceStrategy::DynamicQuality => self.dynamic_quality_inference(input, &context)?,
            InferenceStrategy::Cached => {
                return Err(runtime_error("Cache strategy should not reach here"))
            },
        };

        let inference_time = start_time.elapsed();

        // Cache the result if enabled
        if self.config.enable_caching {
            self.cache_manager.put(input, &context, &result)?;
        }

        // Update performance statistics
        self.update_performance_stats(&strategy, inference_time, &result);

        // Update performance predictor
        self.performance_predictor.update(
            &capabilities,
            &strategy,
            inference_time,
            result.quality_score,
        );

        Ok(result)
    }

    /// Select optimal inference strategy based on device capabilities and context
    fn select_strategy(
        &self,
        capabilities: &DeviceCapabilities,
        context: &InferenceContext,
    ) -> Result<InferenceStrategy> {
        let mut strategy_scores = HashMap::new();

        // Score each strategy based on current conditions
        strategy_scores.insert(
            InferenceStrategy::Full,
            self.score_full_strategy(capabilities, context),
        );
        strategy_scores.insert(
            InferenceStrategy::EarlyExit,
            self.score_early_exit_strategy(capabilities, context),
        );
        strategy_scores.insert(
            InferenceStrategy::Progressive,
            self.score_progressive_strategy(capabilities, context),
        );
        strategy_scores.insert(
            InferenceStrategy::Cascade,
            self.score_cascade_strategy(capabilities, context),
        );
        strategy_scores.insert(
            InferenceStrategy::DynamicQuality,
            self.score_dynamic_quality_strategy(capabilities, context),
        );

        // Select strategy with highest score
        let best_strategy = strategy_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| *strategy)
            .unwrap_or(InferenceStrategy::Full);

        Ok(best_strategy)
    }

    /// Score full inference strategy
    fn score_full_strategy(
        &self,
        capabilities: &DeviceCapabilities,
        context: &InferenceContext,
    ) -> f32 {
        let mut score = 1.0;

        // Reduce score if device is constrained
        if capabilities.battery_level < 0.3 {
            score *= 0.5;
        }
        if capabilities.thermal_state == ThermalState::Hot
            || capabilities.thermal_state == ThermalState::Critical
        {
            score *= 0.3;
        }
        if capabilities.available_memory_mb < 500 {
            score *= 0.4;
        }

        // Increase score if high quality is required
        if context.quality_requirement > 0.9 {
            score *= 1.5;
        }

        score
    }

    /// Score early exit strategy
    fn score_early_exit_strategy(
        &self,
        capabilities: &DeviceCapabilities,
        context: &InferenceContext,
    ) -> f32 {
        let mut score = 0.8;

        // Increase score if device is constrained
        if capabilities.battery_level < 0.5 {
            score *= 1.3;
        }
        if capabilities.thermal_state == ThermalState::Warm
            || capabilities.thermal_state == ThermalState::Hot
        {
            score *= 1.4;
        }
        if context.latency_requirement < Duration::from_millis(50) {
            score *= 1.5;
        }

        // Reduce score if high quality is required
        if context.quality_requirement > 0.85 {
            score *= 0.6;
        }

        score
    }

    /// Score progressive inference strategy
    fn score_progressive_strategy(
        &self,
        capabilities: &DeviceCapabilities,
        context: &InferenceContext,
    ) -> f32 {
        let mut score = 0.7;

        // Increase score for interactive applications
        if context.is_interactive {
            score *= 1.4;
        }

        // Adjust based on network state
        if capabilities.network_state == NetworkState::Cellular {
            score *= 1.2;
        }

        score
    }

    /// Score cascade inference strategy
    fn score_cascade_strategy(
        &self,
        capabilities: &DeviceCapabilities,
        context: &InferenceContext,
    ) -> f32 {
        let mut score = 0.6;

        // Increase score if variable quality is acceptable
        if context.quality_requirement < 0.8 {
            score *= 1.3;
        }

        // Increase score for power-constrained scenarios
        if capabilities.power_source == PowerSource::Battery && capabilities.battery_level < 0.4 {
            score *= 1.4;
        }

        score
    }

    /// Score dynamic quality strategy
    fn score_dynamic_quality_strategy(
        &self,
        capabilities: &DeviceCapabilities,
        context: &InferenceContext,
    ) -> f32 {
        let mut score = 0.9;

        // Always good option for mobile deployment
        if capabilities.thermal_state != ThermalState::Normal {
            score *= 1.2;
        }
        if capabilities.available_memory_mb < 1000 {
            score *= 1.1;
        }

        score
    }

    /// Perform full model inference
    fn full_inference(
        &mut self,
        input: &Tensor,
        context: &InferenceContext,
    ) -> Result<InferenceResult> {
        // Simulate full model inference
        let output = self.run_full_model(input)?;

        Ok(InferenceResult {
            output,
            strategy_used: InferenceStrategy::Full,
            quality_score: 1.0,
            confidence: 0.95,
            computation_cost: 1.0,
            exit_layer: None,
        })
    }

    /// Perform early exit inference
    fn early_exit_inference(
        &mut self,
        input: &Tensor,
        context: &InferenceContext,
    ) -> Result<InferenceResult> {
        // Simulate early exit inference
        let (output, exit_layer, quality) = self.run_early_exit_model(input, context)?;

        Ok(InferenceResult {
            output,
            strategy_used: InferenceStrategy::EarlyExit,
            quality_score: quality,
            confidence: quality * 0.9,
            computation_cost: exit_layer as f32 / 12.0, // Assuming 12 layers total
            exit_layer: Some(exit_layer),
        })
    }

    /// Perform progressive inference
    fn progressive_inference(
        &mut self,
        input: &Tensor,
        context: &InferenceContext,
    ) -> Result<InferenceResult> {
        // Start with low quality and progressively improve
        let mut current_quality = 0.5;
        let mut output = self.run_partial_model(input, current_quality)?;

        // Progressively improve if needed and time allows
        while current_quality < context.quality_requirement && current_quality < 1.0 {
            current_quality += 0.1;
            output = self.run_partial_model(input, current_quality)?;
        }

        Ok(InferenceResult {
            output,
            strategy_used: InferenceStrategy::Progressive,
            quality_score: current_quality,
            confidence: current_quality * 0.85,
            computation_cost: current_quality,
            exit_layer: None,
        })
    }

    /// Perform cascade inference
    fn cascade_inference(
        &mut self,
        input: &Tensor,
        context: &InferenceContext,
    ) -> Result<InferenceResult> {
        // Try simple model first
        let simple_result = self.run_simple_model(input)?;

        // Check if simple result is sufficient
        if simple_result.confidence > context.quality_requirement {
            return Ok(simple_result);
        }

        // Fall back to complex model
        let complex_result = self.run_complex_model(input)?;
        Ok(complex_result)
    }

    /// Perform dynamic quality inference
    fn dynamic_quality_inference(
        &mut self,
        input: &Tensor,
        context: &InferenceContext,
    ) -> Result<InferenceResult> {
        // Determine optimal quality level based on constraints
        let target_quality = self.quality_controller.determine_target_quality(
            &self
                .device_monitor
                .lock()
                .expect("device_monitor lock should not be poisoned")
                .get_capabilities()?,
            context,
        )?;

        let output = self.run_quality_adjusted_model(input, target_quality)?;

        Ok(InferenceResult {
            output,
            strategy_used: InferenceStrategy::DynamicQuality,
            quality_score: target_quality,
            confidence: target_quality * 0.9,
            computation_cost: target_quality,
            exit_layer: None,
        })
    }

    /// Run full model (placeholder)
    fn run_full_model(&self, input: &Tensor) -> Result<Tensor> {
        // Placeholder - would run complete model
        Ok(input.clone())
    }

    /// Run early exit model (placeholder)
    fn run_early_exit_model(
        &self,
        input: &Tensor,
        context: &InferenceContext,
    ) -> Result<(Tensor, usize, f32)> {
        // Placeholder - would run model with early exit capability
        let exit_layer =
            if context.latency_requirement < Duration::from_millis(50) { 6 } else { 10 };
        let quality = exit_layer as f32 / 12.0 + 0.3;
        Ok((input.clone(), exit_layer, quality))
    }

    /// Run partial model (placeholder)
    fn run_partial_model(&self, input: &Tensor, quality: f32) -> Result<Tensor> {
        // Placeholder - would run model with specified quality level
        Ok(input.clone())
    }

    /// Run simple model (placeholder)
    fn run_simple_model(&self, input: &Tensor) -> Result<InferenceResult> {
        Ok(InferenceResult {
            output: input.clone(),
            strategy_used: InferenceStrategy::Cascade,
            quality_score: 0.6,
            confidence: 0.6,
            computation_cost: 0.3,
            exit_layer: None,
        })
    }

    /// Run complex model (placeholder)
    fn run_complex_model(&self, input: &Tensor) -> Result<InferenceResult> {
        Ok(InferenceResult {
            output: input.clone(),
            strategy_used: InferenceStrategy::Cascade,
            quality_score: 0.95,
            confidence: 0.95,
            computation_cost: 1.0,
            exit_layer: None,
        })
    }

    /// Run quality-adjusted model (placeholder)
    fn run_quality_adjusted_model(&self, input: &Tensor, quality: f32) -> Result<Tensor> {
        // Placeholder - would run model with dynamic quality adjustment
        Ok(input.clone())
    }

    /// Update performance statistics
    fn update_performance_stats(
        &mut self,
        strategy: &InferenceStrategy,
        duration: Duration,
        result: &InferenceResult,
    ) {
        self.stats.total_inferences += 1;
        self.stats.total_inference_time += duration;

        match strategy {
            InferenceStrategy::Full => self.stats.full_inferences += 1,
            InferenceStrategy::EarlyExit => self.stats.early_exit_inferences += 1,
            InferenceStrategy::Progressive => self.stats.progressive_inferences += 1,
            InferenceStrategy::Cascade => self.stats.cascade_inferences += 1,
            InferenceStrategy::DynamicQuality => self.stats.dynamic_quality_inferences += 1,
            InferenceStrategy::Cached => self.stats.cache_hits += 1,
        }

        self.stats.avg_quality_score = (self.stats.avg_quality_score
            * (self.stats.total_inferences - 1) as f32
            + result.quality_score)
            / self.stats.total_inferences as f32;
        self.stats.avg_computation_cost = (self.stats.avg_computation_cost
            * (self.stats.total_inferences - 1) as f32
            + result.computation_cost)
            / self.stats.total_inferences as f32;
    }

    /// Get adaptive inference statistics
    pub fn get_stats(&self) -> &AdaptiveStats {
        &self.stats
    }
}

/// Device monitoring system
struct DeviceMonitor {
    capabilities: DeviceCapabilities,
    last_update: Instant,
}

impl DeviceMonitor {
    fn new() -> Result<Self> {
        Ok(Self {
            capabilities: DeviceCapabilities {
                cpu_performance: 0.8,
                gpu_performance: 0.7,
                available_memory_mb: 2048,
                battery_level: 0.8,
                thermal_state: ThermalState::Normal,
                power_source: PowerSource::Battery,
                network_state: NetworkState::WiFi,
            },
            last_update: Instant::now(),
        })
    }

    fn get_capabilities(&mut self) -> Result<DeviceCapabilities> {
        // Update capabilities if needed (every 5 seconds)
        if self.last_update.elapsed() > Duration::from_secs(5) {
            self.update_capabilities()?;
            self.last_update = Instant::now();
        }
        Ok(self.capabilities.clone())
    }

    fn update_capabilities(&mut self) -> Result<()> {
        // Placeholder - would query actual device status
        Ok(())
    }
}

/// Performance prediction system
struct PerformancePredictor {
    strategy_performance: HashMap<InferenceStrategy, StrategyMetrics>,
}

impl PerformancePredictor {
    fn new() -> Self {
        Self {
            strategy_performance: HashMap::new(),
        }
    }

    fn update(
        &mut self,
        capabilities: &DeviceCapabilities,
        strategy: &InferenceStrategy,
        duration: Duration,
        quality: f32,
    ) {
        let metrics = self.strategy_performance.entry(*strategy).or_default();
        metrics.update(duration, quality);
    }

    fn predict_performance(
        &self,
        capabilities: &DeviceCapabilities,
        strategy: &InferenceStrategy,
    ) -> Option<(Duration, f32)> {
        self.strategy_performance
            .get(strategy)
            .map(|metrics| (metrics.avg_duration, metrics.avg_quality))
    }
}

/// Quality control system
struct QualityController {
    min_threshold: f32,
}

impl QualityController {
    fn new(min_threshold: f32) -> Self {
        Self { min_threshold }
    }

    fn determine_target_quality(
        &self,
        capabilities: &DeviceCapabilities,
        context: &InferenceContext,
    ) -> Result<f32> {
        let mut target_quality = context.quality_requirement;

        // Adjust based on device constraints
        if capabilities.battery_level < 0.3 {
            target_quality *= 0.8;
        }
        if capabilities.thermal_state == ThermalState::Hot {
            target_quality *= 0.7;
        }
        if capabilities.available_memory_mb < 500 {
            target_quality *= 0.9;
        }

        // Ensure minimum threshold
        target_quality = target_quality.max(self.min_threshold);

        Ok(target_quality)
    }
}

/// Cache management system
struct CacheManager {
    enabled: bool,
    cache: HashMap<String, CacheEntry>,
    max_entries: usize,
}

impl CacheManager {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            cache: HashMap::new(),
            max_entries: 100,
        }
    }

    fn get(&self, input: &Tensor, context: &InferenceContext) -> Result<Option<InferenceResult>> {
        if !self.enabled {
            return Ok(None);
        }

        let key = self.compute_cache_key(input, context)?;
        Ok(self.cache.get(&key).map(|entry| entry.result.clone()))
    }

    fn put(
        &mut self,
        input: &Tensor,
        context: &InferenceContext,
        result: &InferenceResult,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let key = self.compute_cache_key(input, context)?;

        // Evict old entries if needed
        if self.cache.len() >= self.max_entries {
            self.evict_oldest();
        }

        self.cache.insert(
            key,
            CacheEntry {
                result: result.clone(),
                timestamp: Instant::now(),
            },
        );

        Ok(())
    }

    fn compute_cache_key(&self, input: &Tensor, context: &InferenceContext) -> Result<String> {
        // Simplified cache key - would use proper hashing in production
        Ok(format!(
            "{}_{}",
            input.shape().iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x"),
            context.quality_requirement
        ))
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(key, _)| key.clone())
        {
            self.cache.remove(&oldest_key);
        }
    }
}

/// Batch scheduling system
struct BatchScheduler {
    enabled: bool,
    pending_requests: Vec<BatchRequest>,
    max_batch_size: usize,
    max_wait_time: Duration,
}

impl BatchScheduler {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pending_requests: Vec::new(),
            max_batch_size: 8,
            max_wait_time: Duration::from_millis(10),
        }
    }
}

/// Inference context
#[derive(Debug, Clone)]
pub struct InferenceContext {
    /// Required quality (0.0 - 1.0)
    pub quality_requirement: f32,
    /// Maximum latency tolerance
    pub latency_requirement: Duration,
    /// Whether this is an interactive request
    pub is_interactive: bool,
    /// Priority level
    pub priority: Priority,
}

/// Priority levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

/// Inference result
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Model output
    pub output: Tensor,
    /// Strategy used for inference
    pub strategy_used: InferenceStrategy,
    /// Quality score achieved
    pub quality_score: f32,
    /// Confidence in the result
    pub confidence: f32,
    /// Relative computation cost
    pub computation_cost: f32,
    /// Exit layer if early exit was used
    pub exit_layer: Option<usize>,
}

/// Cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    result: InferenceResult,
    timestamp: Instant,
}

/// Batch request
#[derive(Debug)]
struct BatchRequest {
    input: Tensor,
    context: InferenceContext,
    timestamp: Instant,
}

/// Strategy performance metrics
#[derive(Debug, Default)]
struct StrategyMetrics {
    total_calls: usize,
    total_duration: Duration,
    total_quality: f32,
    avg_duration: Duration,
    avg_quality: f32,
}

impl StrategyMetrics {
    fn update(&mut self, duration: Duration, quality: f32) {
        self.total_calls += 1;
        self.total_duration += duration;
        self.total_quality += quality;

        self.avg_duration = self.total_duration / self.total_calls as u32;
        self.avg_quality = self.total_quality / self.total_calls as f32;
    }
}

/// Adaptive inference statistics
#[derive(Debug, Clone, Default)]
pub struct AdaptiveStats {
    pub total_inferences: usize,
    pub total_inference_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub full_inferences: usize,
    pub early_exit_inferences: usize,
    pub progressive_inferences: usize,
    pub cascade_inferences: usize,
    pub dynamic_quality_inferences: usize,
    pub avg_quality_score: f32,
    pub avg_computation_cost: f32,
}

impl AdaptiveStats {
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32
        }
    }

    /// Get average inference time
    pub fn avg_inference_time(&self) -> Duration {
        if self.total_inferences == 0 {
            Duration::from_millis(0)
        } else {
            self.total_inference_time / self.total_inferences as u32
        }
    }

    /// Get strategy distribution
    pub fn strategy_distribution(&self) -> HashMap<InferenceStrategy, f32> {
        let mut distribution = HashMap::new();
        let total = self.total_inferences as f32;

        if total > 0.0 {
            distribution.insert(InferenceStrategy::Full, self.full_inferences as f32 / total);
            distribution.insert(
                InferenceStrategy::EarlyExit,
                self.early_exit_inferences as f32 / total,
            );
            distribution.insert(
                InferenceStrategy::Progressive,
                self.progressive_inferences as f32 / total,
            );
            distribution.insert(
                InferenceStrategy::Cascade,
                self.cascade_inferences as f32 / total,
            );
            distribution.insert(
                InferenceStrategy::DynamicQuality,
                self.dynamic_quality_inferences as f32 / total,
            );
        }

        distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveConfig::default();
        assert!(config.dynamic_quality);
        assert!(config.early_exit);
        assert_eq!(config.min_quality_threshold, 0.7);
    }

    #[test]
    fn test_adaptive_inference_engine_creation() {
        let config = AdaptiveConfig::default();
        let engine = AdaptiveInferenceEngine::new(config, MobileBackend::CPU);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_adaptive_stats() {
        let mut stats = AdaptiveStats::default();
        stats.cache_hits = 80;
        stats.cache_misses = 20;

        assert_eq!(stats.cache_hit_rate(), 0.8);
    }

    #[test]
    fn test_device_capabilities() {
        let capabilities = DeviceCapabilities {
            cpu_performance: 0.8,
            gpu_performance: 0.7,
            available_memory_mb: 2048,
            battery_level: 0.8,
            thermal_state: ThermalState::Normal,
            power_source: PowerSource::Battery,
            network_state: NetworkState::WiFi,
        };

        assert_eq!(capabilities.thermal_state, ThermalState::Normal);
        assert_eq!(capabilities.power_source, PowerSource::Battery);
    }
}
