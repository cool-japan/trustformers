//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, RwLock};
use tokio_stream::wrappers::BroadcastStream;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::functions::AggregationRule;

/// Intelligent buffer management system
#[derive(Debug)]
pub struct IntelligentBufferManager {
    /// Buffer configuration
    pub config: IntelligentBufferingConfig,
    /// Current buffer sizes per stream
    pub buffer_sizes: HashMap<Uuid, usize>,
    /// Buffer utilization tracking
    pub utilization_history: Vec<BufferUtilization>,
    /// Memory pressure tracking
    pub memory_pressure: f32,
    /// Performance predictions
    pub performance_predictor: Option<BufferPerformancePredictor>,
}
impl IntelligentBufferManager {
    pub fn new(config: IntelligentBufferingConfig) -> Self {
        Self {
            config,
            buffer_sizes: HashMap::new(),
            utilization_history: Vec::new(),
            memory_pressure: 0.0,
            performance_predictor: None,
        }
    }
    pub async fn calculate_optimal_buffer_size(&self) -> Result<usize> {
        if !self.config.enable_intelligent_buffering {
            return Ok(self.config.max_buffer_size);
        }
        match self.config.buffer_strategy {
            BufferStrategy::Fixed => Ok(self.config.max_buffer_size),
            BufferStrategy::Adaptive => self.calculate_adaptive_buffer_size(),
            BufferStrategy::LoadBased => self.calculate_load_based_buffer_size(),
            BufferStrategy::LatencyBased => self.calculate_latency_based_buffer_size(),
            BufferStrategy::PredictiveBased => self.calculate_predictive_buffer_size(),
        }
    }
    fn calculate_adaptive_buffer_size(&self) -> Result<usize> {
        let base_size = (self.config.min_buffer_size + self.config.max_buffer_size) / 2;
        let pressure_factor = 1.0 - self.memory_pressure;
        let adjusted_size = (base_size as f32 * pressure_factor) as usize;
        Ok(adjusted_size.max(self.config.min_buffer_size).min(self.config.max_buffer_size))
    }
    fn calculate_load_based_buffer_size(&self) -> Result<usize> {
        if self.utilization_history.is_empty() {
            return Ok(self.config.max_buffer_size);
        }
        let recent_utilization: f32 = self
            .utilization_history
            .iter()
            .rev()
            .take(10)
            .map(|u| u.utilization)
            .sum::<f32>()
            / 10.0;
        let size_factor = if recent_utilization > self.config.utilization_threshold {
            self.config.adjustment_factor
        } else {
            1.0 / self.config.adjustment_factor
        };
        let base_size = (self.config.min_buffer_size + self.config.max_buffer_size) / 2;
        let adjusted_size = (base_size as f32 * size_factor) as usize;
        Ok(adjusted_size.max(self.config.min_buffer_size).min(self.config.max_buffer_size))
    }
    fn calculate_latency_based_buffer_size(&self) -> Result<usize> {
        let target_latency = 100.0;
        let current_latency =
            self.utilization_history.last().map(|u| u.latency_impact).unwrap_or(50.0);
        let latency_ratio = target_latency / current_latency.max(1.0);
        let base_size = (self.config.min_buffer_size + self.config.max_buffer_size) / 2;
        let adjusted_size = (base_size as f32 * latency_ratio) as usize;
        Ok(adjusted_size.max(self.config.min_buffer_size).min(self.config.max_buffer_size))
    }
    fn calculate_predictive_buffer_size(&self) -> Result<usize> {
        if let Some(predictor) = &self.performance_predictor {
            let optimal_size = predictor.predict_optimal_size()?;
            Ok(optimal_size.max(self.config.min_buffer_size).min(self.config.max_buffer_size))
        } else {
            self.calculate_adaptive_buffer_size()
        }
    }
    pub async fn optimize_buffers(&mut self) {
        self.update_memory_pressure().await;
        self.record_utilization().await;
        let current_optimal = self.calculate_optimal_buffer_size().await.unwrap_or(1000);
        for (stream_id, current_size) in &mut self.buffer_sizes {
            let difference = (current_optimal as f32 - *current_size as f32).abs();
            if difference > (*current_size as f32 * 0.1) {
                *current_size = current_optimal;
                debug!(
                    "Optimized buffer size for stream {}: {}",
                    stream_id, current_optimal
                );
            }
        }
    }
    async fn update_memory_pressure(&mut self) {
        self.memory_pressure = (thread_rng().random::<f32>() * 0.3).max(0.0);
    }
    async fn record_utilization(&mut self) {
        let mut rng = thread_rng();
        for (stream_id, buffer_size) in &self.buffer_sizes {
            let utilization = BufferUtilization {
                timestamp: Instant::now(),
                stream_id: *stream_id,
                utilization: rng.random::<f32>(),
                latency_impact: rng.random::<f32>() * 100.0,
                memory_usage: *buffer_size * 100,
            };
            self.utilization_history.push(utilization);
            if self.utilization_history.len() > 1000 {
                self.utilization_history.remove(0);
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct EventPattern {
    pub event_types: Vec<String>,
    pub severity_threshold: Option<f32>,
    pub frequency_threshold: Option<f32>,
    pub custom_conditions: HashMap<String, String>,
}
/// Filtering options for stream events
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamFilter {
    /// Filter by session IDs
    pub session_ids: Option<Vec<Uuid>>,
    /// Filter by event types
    pub event_types: Option<Vec<String>>,
    /// Filter by minimum severity level
    pub min_severity: Option<AnomalySeverity>,
    /// Filter by time range
    pub time_range: Option<(SystemTime, SystemTime)>,
    /// Custom filter expressions
    pub custom_filters: HashMap<String, String>,
}
/// Stream control event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamControlType {
    StreamStarted,
    StreamStopped,
    StreamPaused,
    StreamResumed,
    StreamError,
    ConnectionEstablished,
    ConnectionLost,
}
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub average_bandwidth: u64,
    pub average_latency: u64,
    pub connection_stability: f32,
    pub quality_score_trend: f32,
}
/// Rate limiting for stream events
#[derive(Debug)]
struct RateLimiter {
    tokens: u32,
    last_refill: Instant,
    max_tokens: u32,
    #[allow(dead_code)]
    refill_rate: u32,
}
impl RateLimiter {
    fn new(max_rate_per_second: u32) -> Self {
        Self {
            tokens: max_rate_per_second,
            last_refill: Instant::now(),
            max_tokens: max_rate_per_second,
            refill_rate: max_rate_per_second,
        }
    }
    fn try_consume(&mut self) -> bool {
        self.refill_tokens();
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }
    fn refill_tokens(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        if elapsed >= Duration::from_secs(1) {
            self.tokens = self.max_tokens;
            self.last_refill = now;
        }
    }
}
/// Event prioritization system
#[derive(Debug)]
pub struct EventPrioritizer {
    /// Priority rules
    pub priority_rules: Vec<PriorityRule>,
    /// Event importance scoring
    pub importance_scorer: ImportanceScorer,
    /// Priority queues
    pub priority_queues: HashMap<EventPriority, Vec<StreamEvent>>,
}
impl EventPrioritizer {
    pub fn new() -> Self {
        Self {
            priority_rules: Vec::new(),
            importance_scorer: ImportanceScorer::new(),
            priority_queues: HashMap::new(),
        }
    }
    pub async fn calculate_importance(&self, event: &StreamEvent) -> Result<f32> {
        self.importance_scorer.calculate_importance(event).await
    }
    pub async fn process_priority_queues(&mut self) {
        for priority in [
            EventPriority::Critical,
            EventPriority::High,
            EventPriority::Medium,
            EventPriority::Low,
            EventPriority::Background,
        ] {
            if let Some(queue) = self.priority_queues.get_mut(&priority) {
                while let Some(_event) = queue.pop() {
                    debug!("Processing {:?} priority event", priority);
                    break;
                }
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct BufferPerformancePoint {
    pub buffer_size: usize,
    pub throughput: f64,
    pub latency: f64,
    pub memory_usage: usize,
    pub timestamp: Instant,
}
#[derive(Debug, Clone)]
pub struct AggregationMetrics {
    pub windows_processed: u64,
    pub average_window_size: f32,
    pub aggregation_latency: f64,
    pub data_reduction_ratio: f32,
}
#[derive(Debug, Clone)]
pub struct PriorityRule {
    pub rule_id: String,
    pub event_pattern: EventPattern,
    pub priority: EventPriority,
    pub score_modifier: f32,
}
/// Enhanced streaming debugger with advanced capabilities
#[derive(Debug)]
pub struct EnhancedStreamingDebugger {
    /// Base streaming debugger
    base: StreamingDebugger,
    /// Adaptive streaming configuration
    adaptive_config: AdaptiveStreamingConfig,
    /// Real-time aggregation configuration
    aggregation_config: RealTimeAggregationConfig,
    /// Intelligent buffering configuration
    buffering_config: IntelligentBufferingConfig,
    /// Network condition monitor
    network_monitor: Arc<RwLock<NetworkConditionMonitor>>,
    /// Real-time aggregator
    aggregator: Arc<RwLock<RealTimeAggregator>>,
    /// Intelligent buffer manager
    buffer_manager: Arc<RwLock<IntelligentBufferManager>>,
    /// Performance metrics
    enhanced_metrics: Arc<RwLock<EnhancedStreamMetrics>>,
    /// Event prioritization system
    prioritizer: Arc<RwLock<EventPrioritizer>>,
}
impl EnhancedStreamingDebugger {
    /// Create a new enhanced streaming debugger
    pub fn new(
        base_config: StreamingDebugConfig,
        adaptive_config: AdaptiveStreamingConfig,
        aggregation_config: RealTimeAggregationConfig,
        buffering_config: IntelligentBufferingConfig,
    ) -> Self {
        let base = StreamingDebugger::new(base_config);
        Self {
            base,
            adaptive_config,
            aggregation_config: aggregation_config.clone(),
            buffering_config: buffering_config.clone(),
            network_monitor: Arc::new(RwLock::new(NetworkConditionMonitor::new())),
            aggregator: Arc::new(RwLock::new(RealTimeAggregator::new(aggregation_config))),
            buffer_manager: Arc::new(RwLock::new(IntelligentBufferManager::new(buffering_config))),
            enhanced_metrics: Arc::new(RwLock::new(EnhancedStreamMetrics::default())),
            prioritizer: Arc::new(RwLock::new(EventPrioritizer::new())),
        }
    }
    /// Start enhanced streaming with adaptive capabilities
    pub async fn start_enhanced_streaming(&mut self) -> Result<()> {
        self.base.start_streaming().await?;
        self.start_network_monitoring().await?;
        self.start_real_time_aggregation().await?;
        self.start_intelligent_buffering().await?;
        self.start_event_prioritization().await?;
        info!("Enhanced streaming debugger started successfully");
        Ok(())
    }
    /// Stop enhanced streaming
    pub async fn stop_enhanced_streaming(&mut self) -> Result<()> {
        self.base.stop_streaming().await?;
        info!("Enhanced streaming debugger stopped");
        Ok(())
    }
    /// Stream events with adaptive quality
    pub async fn stream_adaptive(&mut self, event: StreamEvent) -> Result<()> {
        let network_monitor = self.network_monitor.read().await;
        let quality_score = network_monitor.quality_score;
        drop(network_monitor);
        let adjusted_event = self.adjust_event_quality(event, quality_score).await?;
        let prioritized_event = self.prioritize_event(adjusted_event).await?;
        self.stream_with_intelligent_buffering(prioritized_event).await?;
        Ok(())
    }
    /// Adjust event quality based on network conditions
    async fn adjust_event_quality(
        &self,
        mut event: StreamEvent,
        quality_score: f32,
    ) -> Result<StreamEvent> {
        if !self.adaptive_config.enable_adaptive_streaming {
            return Ok(event);
        }
        let adjusted_quality = (quality_score * self.adaptive_config.max_quality)
            .max(self.adaptive_config.min_quality)
            .min(self.adaptive_config.max_quality);
        match &mut event {
            StreamEvent::TensorData { values, .. } => {
                if adjusted_quality < 1.0 {
                    let sample_rate = adjusted_quality as usize;
                    if sample_rate > 0 && values.len() > sample_rate {
                        values.truncate(values.len() / sample_rate + 1);
                    }
                }
            },
            StreamEvent::PerformanceMetrics { .. } => {},
            _ => {},
        }
        Ok(event)
    }
    /// Prioritize events based on importance
    async fn prioritize_event(&self, event: StreamEvent) -> Result<StreamEvent> {
        let prioritizer = self.prioritizer.read().await;
        let _importance_score = prioritizer.calculate_importance(&event).await?;
        drop(prioritizer);
        Ok(event)
    }
    /// Stream with intelligent buffering
    async fn stream_with_intelligent_buffering(&mut self, event: StreamEvent) -> Result<()> {
        let buffer_manager = self.buffer_manager.write().await;
        let _optimal_buffer_size = buffer_manager.calculate_optimal_buffer_size().await?;
        drop(buffer_manager);
        self.base.stream_event(event).await?;
        Ok(())
    }
    /// Start network condition monitoring
    async fn start_network_monitoring(&self) -> Result<()> {
        let network_monitor = self.network_monitor.clone();
        let monitoring_interval =
            Duration::from_millis(self.adaptive_config.monitoring_interval_ms);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitoring_interval);
            loop {
                interval.tick().await;
                let mut monitor = network_monitor.write().await;
                monitor.update_conditions().await;
                drop(monitor);
            }
        });
        Ok(())
    }
    /// Start real-time aggregation
    async fn start_real_time_aggregation(&self) -> Result<()> {
        let aggregator = self.aggregator.clone();
        let window_size = Duration::from_secs(self.aggregation_config.window_size_seconds);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(window_size);
            loop {
                interval.tick().await;
                let mut agg = aggregator.write().await;
                agg.process_windows().await;
                drop(agg);
            }
        });
        Ok(())
    }
    /// Start intelligent buffering
    async fn start_intelligent_buffering(&self) -> Result<()> {
        let buffer_manager = self.buffer_manager.clone();
        let monitoring_interval = Duration::from_millis(1000);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitoring_interval);
            loop {
                interval.tick().await;
                let mut manager = buffer_manager.write().await;
                manager.optimize_buffers().await;
                drop(manager);
            }
        });
        Ok(())
    }
    /// Start event prioritization
    async fn start_event_prioritization(&self) -> Result<()> {
        let prioritizer = self.prioritizer.clone();
        let processing_interval = Duration::from_millis(100);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(processing_interval);
            loop {
                interval.tick().await;
                let mut p = prioritizer.write().await;
                p.process_priority_queues().await;
                drop(p);
            }
        });
        Ok(())
    }
    /// Get comprehensive streaming metrics
    pub async fn get_enhanced_metrics(&self) -> EnhancedStreamMetrics {
        self.enhanced_metrics.read().await.clone()
    }
    /// Configure adaptive streaming parameters
    pub async fn configure_adaptive_streaming(
        &mut self,
        config: AdaptiveStreamingConfig,
    ) -> Result<()> {
        self.adaptive_config = config;
        info!("Adaptive streaming configuration updated");
        Ok(())
    }
    /// Configure real-time aggregation
    pub async fn configure_aggregation(&mut self, config: RealTimeAggregationConfig) -> Result<()> {
        self.aggregation_config = config.clone();
        let mut aggregator = self.aggregator.write().await;
        aggregator.config = config;
        Ok(())
    }
    /// Configure intelligent buffering
    pub async fn configure_buffering(&mut self, config: IntelligentBufferingConfig) -> Result<()> {
        self.buffering_config = config.clone();
        let mut buffer_manager = self.buffer_manager.write().await;
        buffer_manager.config = config;
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub enum MLModelType {
    LinearRegression,
    RandomForest,
    NeuralNetwork,
    GradientBoosting,
}
/// Supported streaming data formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StreamFormat {
    Json,
    MessagePack,
    Protobuf,
    Avro,
}
/// Types of anomalies that can be detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnomalyType {
    GradientExplosion,
    GradientVanishing,
    NumericalInstability,
    MemoryLeak,
    PerformanceDegradation,
    LossSpike,
    TrainingStagnation,
    ResourceExhaustion,
}
#[derive(Debug, Clone)]
pub struct BufferUtilization {
    pub timestamp: Instant,
    pub stream_id: Uuid,
    pub utilization: f32,
    pub latency_impact: f32,
    pub memory_usage: usize,
}
/// Enhanced streaming metrics
#[derive(Debug, Clone)]
pub struct EnhancedStreamMetrics {
    /// Base metrics
    pub base_metrics: StreamMetrics,
    /// Adaptive streaming metrics
    pub adaptive_metrics: AdaptiveStreamingMetrics,
    /// Aggregation metrics
    pub aggregation_metrics: AggregationMetrics,
    /// Buffer performance metrics
    pub buffer_metrics: BufferMetrics,
    /// Network condition metrics
    pub network_metrics: NetworkMetrics,
}
#[derive(Debug, Clone)]
pub struct ImportanceScore {
    pub event_id: Uuid,
    pub score: f32,
    pub factors: HashMap<String, f32>,
    pub timestamp: Instant,
}
/// Real-time data aggregator
pub struct RealTimeAggregator {
    /// Current aggregation windows
    pub windows: HashMap<String, AggregationWindow>,
    /// Aggregation configuration
    pub config: RealTimeAggregationConfig,
    /// Custom aggregation rules
    pub custom_rules: HashMap<String, Box<dyn AggregationRule + Send + Sync>>,
}
impl RealTimeAggregator {
    pub fn new(config: RealTimeAggregationConfig) -> Self {
        Self {
            windows: HashMap::new(),
            config,
            custom_rules: HashMap::new(),
        }
    }
    pub async fn process_windows(&mut self) {
        let now = Instant::now();
        let window_duration = Duration::from_secs(self.config.window_size_seconds);
        let mut completed_windows = Vec::new();
        for (window_id, window) in &self.windows {
            if now.duration_since(window.start_time) >= window_duration {
                completed_windows.push(window_id.clone());
            }
        }
        for window_id in completed_windows {
            if let Some(mut window) = self.windows.remove(&window_id) {
                self.aggregate_window(&mut window).await;
                debug!(
                    "Aggregated window {} with {} events",
                    window_id,
                    window.events.len()
                );
            }
        }
    }
    async fn aggregate_window(&self, window: &mut AggregationWindow) {
        for function in &self.config.aggregation_functions {
            if let Ok(result) = self.apply_aggregation_function(function, &window.events) {
                window.aggregated_results.insert(function.clone(), result);
            }
        }
    }
    fn apply_aggregation_function(
        &self,
        function: &AggregationFunction,
        events: &[StreamEvent],
    ) -> Result<f64> {
        match function {
            AggregationFunction::Count => Ok(events.len() as f64),
            AggregationFunction::Mean => {
                let values = self.extract_numeric_values(events);
                if values.is_empty() {
                    Ok(0.0)
                } else {
                    Ok(values.iter().sum::<f64>() / values.len() as f64)
                }
            },
            AggregationFunction::Max => {
                let values = self.extract_numeric_values(events);
                Ok(values.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            },
            AggregationFunction::Min => {
                let values = self.extract_numeric_values(events);
                Ok(values.iter().cloned().fold(f64::INFINITY, f64::min))
            },
            AggregationFunction::Sum => {
                let values = self.extract_numeric_values(events);
                Ok(values.iter().sum())
            },
            AggregationFunction::Rate => {
                let duration = self.config.window_size_seconds as f64;
                Ok(events.len() as f64 / duration)
            },
            _ => Ok(0.0),
        }
    }
    fn extract_numeric_values(&self, events: &[StreamEvent]) -> Vec<f64> {
        let mut values = Vec::new();
        for event in events {
            match event {
                StreamEvent::PerformanceMetrics { latency_ms, .. } => values.push(*latency_ms),
                StreamEvent::TrainingDynamics { loss, .. } => values.push(*loss),
                StreamEvent::TensorData { statistics, .. } => values.push(statistics.mean),
                _ => {},
            }
        }
        values
    }
}
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingMetrics {
    pub quality_adjustments: u64,
    pub average_quality: f32,
    pub bandwidth_utilization: f32,
    pub quality_stability_score: f32,
}
#[derive(Debug, Clone)]
pub struct NetworkMeasurement {
    pub timestamp: Instant,
    pub bandwidth: u64,
    pub latency: u64,
    pub packet_loss: f32,
}
#[derive(Debug, Clone)]
pub struct TrainingDataPoint {
    pub features: Vec<f64>,
    pub importance_score: f32,
    pub user_feedback: Option<f32>,
}
/// Gradient statistics for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStatistics {
    pub l1_norm: f64,
    pub l2_norm: f64,
    pub max_grad: f64,
    pub min_grad: f64,
    pub dead_neuron_ratio: f64,
    pub gradient_diversity: f64,
}
#[derive(Debug, Clone)]
pub struct BufferMetrics {
    pub buffer_adjustments: u64,
    pub average_utilization: f32,
    pub memory_efficiency: f32,
    pub buffer_overflow_count: u64,
}
/// Configuration for streaming debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDebugConfig {
    /// Enable real-time streaming
    pub enable_streaming: bool,
    /// Streaming interval in milliseconds
    pub stream_interval_ms: u64,
    /// Maximum number of concurrent streams
    pub max_concurrent_streams: usize,
    /// Buffer size for streaming data
    pub stream_buffer_size: usize,
    /// Enable stream compression
    pub enable_compression: bool,
    /// Stream data formats to support
    pub supported_formats: Vec<StreamFormat>,
    /// Maximum data retention period in seconds
    pub max_retention_seconds: u64,
    /// Enable stream authentication
    pub enable_authentication: bool,
    /// Stream rate limiting (events per second)
    pub rate_limit_per_second: u32,
}
#[derive(Debug)]
pub struct BufferPerformancePredictor {
    /// Historical performance data
    pub performance_history: Vec<BufferPerformancePoint>,
    /// Prediction model parameters
    pub model_params: Vec<f64>,
    /// Prediction accuracy
    pub accuracy: f32,
}
impl BufferPerformancePredictor {
    pub fn predict_optimal_size(&self) -> Result<usize> {
        if self.performance_history.is_empty() {
            return Ok(1000);
        }
        let optimal_point = self.performance_history.iter().max_by(|a, b| {
            let score_a = a.throughput / a.latency.max(1.0);
            let score_b = b.throughput / b.latency.max(1.0);
            score_a.partial_cmp(&score_b).unwrap()
        });
        Ok(optimal_point.map(|p| p.buffer_size).unwrap_or(1000))
    }
}
#[derive(Debug)]
pub struct AggregationWindow {
    pub window_id: String,
    pub start_time: Instant,
    pub end_time: Instant,
    pub events: Vec<StreamEvent>,
    pub aggregated_results: HashMap<AggregationFunction, f64>,
}
/// Tensor statistics for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub sparsity: f64,
}
/// Advanced adaptive streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStreamingConfig {
    /// Enable adaptive streaming based on network conditions
    pub enable_adaptive_streaming: bool,
    /// Minimum streaming quality (0.0 to 1.0)
    pub min_quality: f32,
    /// Maximum streaming quality (0.0 to 1.0)
    pub max_quality: f32,
    /// Bandwidth threshold for quality adjustment (bytes/sec)
    pub bandwidth_threshold: u64,
    /// Latency threshold for quality adjustment (ms)
    pub latency_threshold_ms: u64,
    /// Quality adjustment step size
    pub quality_step: f32,
    /// Network condition monitoring interval (ms)
    pub monitoring_interval_ms: u64,
}
/// Real-time aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeAggregationConfig {
    /// Enable real-time aggregation
    pub enable_aggregation: bool,
    /// Aggregation window size in seconds
    pub window_size_seconds: u64,
    /// Aggregation functions to apply
    pub aggregation_functions: Vec<AggregationFunction>,
    /// Enable sliding window aggregation
    pub enable_sliding_window: bool,
    /// Maximum number of aggregation windows to retain
    pub max_windows: usize,
    /// Enable custom aggregation rules
    pub enable_custom_rules: bool,
}
#[derive(Debug)]
pub struct ImportanceScorer {
    /// Scoring weights for different factors
    pub weights: ScoringWeights,
    /// Historical importance data
    pub history: Vec<ImportanceScore>,
    /// Machine learning model for importance prediction
    pub ml_model: Option<ImportancePredictionModel>,
}
impl ImportanceScorer {
    pub fn new() -> Self {
        Self {
            weights: ScoringWeights::default(),
            history: Vec::new(),
            ml_model: None,
        }
    }
    pub async fn calculate_importance(&self, event: &StreamEvent) -> Result<f32> {
        let mut score = 0.0;
        let recency_score = 1.0;
        score += recency_score * self.weights.recency_weight;
        let severity_score = self.calculate_severity_score(event);
        score += severity_score * self.weights.severity_weight;
        let frequency_score = self.calculate_frequency_score(event);
        score += frequency_score * self.weights.frequency_weight;
        Ok(score.max(0.0).min(1.0))
    }
    fn calculate_severity_score(&self, event: &StreamEvent) -> f32 {
        match event {
            StreamEvent::AnomalyDetected { severity, .. } => match severity {
                AnomalySeverity::Critical => 1.0,
                AnomalySeverity::High => 0.8,
                AnomalySeverity::Medium => 0.6,
                AnomalySeverity::Low => 0.4,
            },
            StreamEvent::SystemHealth { health_score, .. } => (1.0 - health_score) as f32,
            _ => 0.5,
        }
    }
    fn calculate_frequency_score(&self, _event: &StreamEvent) -> f32 {
        0.5
    }
}
#[derive(Debug)]
pub struct ImportancePredictionModel {
    /// Model type
    pub model_type: MLModelType,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Training data
    pub training_data: Vec<TrainingDataPoint>,
    /// Model accuracy
    pub accuracy: f32,
    /// Last training time
    pub last_training: Instant,
}
/// Stream event types for debugging data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    /// Tensor data stream event
    TensorData {
        session_id: Uuid,
        tensor_id: Uuid,
        name: String,
        shape: Vec<usize>,
        values: Vec<f64>,
        statistics: TensorStatistics,
        timestamp: SystemTime,
    },
    /// Gradient flow event
    GradientFlow {
        session_id: Uuid,
        layer_name: String,
        gradient_norm: f64,
        gradient_statistics: GradientStatistics,
        vanishing_risk: f64,
        exploding_risk: f64,
        timestamp: SystemTime,
    },
    /// Performance metrics event
    PerformanceMetrics {
        session_id: Uuid,
        operation_name: String,
        latency_ms: f64,
        memory_usage_mb: f64,
        cpu_utilization: f64,
        gpu_utilization: Option<f64>,
        timestamp: SystemTime,
    },
    /// Training dynamics event
    TrainingDynamics {
        session_id: Uuid,
        epoch: u32,
        step: u64,
        loss: f64,
        learning_rate: f64,
        accuracy: Option<f64>,
        convergence_score: f64,
        timestamp: SystemTime,
    },
    /// Anomaly detection event
    AnomalyDetected {
        session_id: Uuid,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        description: String,
        confidence: f64,
        affected_components: Vec<String>,
        timestamp: SystemTime,
    },
    /// System health event
    SystemHealth {
        session_id: Uuid,
        health_score: f64,
        memory_pressure: f64,
        resource_utilization: HashMap<String, f64>,
        active_warnings: Vec<String>,
        timestamp: SystemTime,
    },
    /// Stream control events
    StreamControl {
        event_type: StreamControlType,
        stream_id: Uuid,
        message: String,
        timestamp: SystemTime,
    },
}
/// Stream subscription handle
pub struct StreamSubscription {
    subscriber_id: Uuid,
    #[allow(dead_code)]
    receiver: BroadcastStream<StreamEvent>,
    filter: StreamFilter,
    format: StreamFormat,
}
impl StreamSubscription {
    fn new(
        subscriber_id: Uuid,
        receiver: broadcast::Receiver<StreamEvent>,
        filter: StreamFilter,
        format: StreamFormat,
    ) -> Self {
        Self {
            subscriber_id,
            receiver: BroadcastStream::new(receiver),
            filter,
            format,
        }
    }
    /// Get the subscriber ID
    pub fn subscriber_id(&self) -> Uuid {
        self.subscriber_id
    }
    /// Get the stream format
    pub fn format(&self) -> &StreamFormat {
        &self.format
    }
    /// Get filtered stream of events
    pub fn stream(&mut self, event_sender: &broadcast::Sender<StreamEvent>) -> FilteredEventStream {
        FilteredEventStream {
            receiver: BroadcastStream::new(event_sender.subscribe()),
            filter: self.filter.clone(),
        }
    }
    /// Check if event matches the filter
    pub(super) fn matches_filter(event: &StreamEvent, filter: &StreamFilter) -> bool {
        if let Some(ref session_ids) = filter.session_ids {
            let event_session_id = match event {
                StreamEvent::TensorData { session_id, .. } => Some(*session_id),
                StreamEvent::GradientFlow { session_id, .. } => Some(*session_id),
                StreamEvent::PerformanceMetrics { session_id, .. } => Some(*session_id),
                StreamEvent::TrainingDynamics { session_id, .. } => Some(*session_id),
                StreamEvent::AnomalyDetected { session_id, .. } => Some(*session_id),
                StreamEvent::SystemHealth { session_id, .. } => Some(*session_id),
                StreamEvent::StreamControl { .. } => None,
            };
            if let Some(event_session_id) = event_session_id {
                if !session_ids.contains(&event_session_id) {
                    return false;
                }
            }
        }
        if let Some(ref event_types) = filter.event_types {
            let event_type = match event {
                StreamEvent::TensorData { .. } => "TensorData",
                StreamEvent::GradientFlow { .. } => "GradientFlow",
                StreamEvent::PerformanceMetrics { .. } => "PerformanceMetrics",
                StreamEvent::TrainingDynamics { .. } => "TrainingDynamics",
                StreamEvent::AnomalyDetected { .. } => "AnomalyDetected",
                StreamEvent::SystemHealth { .. } => "SystemHealth",
                StreamEvent::StreamControl { .. } => "StreamControl",
            };
            if !event_types.contains(&event_type.to_string()) {
                return false;
            }
        }
        if let Some(ref min_severity) = filter.min_severity {
            if let StreamEvent::AnomalyDetected { severity, .. } = event {
                if severity < min_severity {
                    return false;
                }
            }
        }
        if let Some((start, end)) = filter.time_range {
            let event_timestamp = match event {
                StreamEvent::TensorData { timestamp, .. } => *timestamp,
                StreamEvent::GradientFlow { timestamp, .. } => *timestamp,
                StreamEvent::PerformanceMetrics { timestamp, .. } => *timestamp,
                StreamEvent::TrainingDynamics { timestamp, .. } => *timestamp,
                StreamEvent::AnomalyDetected { timestamp, .. } => *timestamp,
                StreamEvent::SystemHealth { timestamp, .. } => *timestamp,
                StreamEvent::StreamControl { timestamp, .. } => *timestamp,
            };
            if event_timestamp < start || event_timestamp > end {
                return false;
            }
        }
        true
    }
}
/// Buffer adjustment strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BufferStrategy {
    Fixed,
    Adaptive,
    PredictiveBased,
    LoadBased,
    LatencyBased,
}
/// Severity levels for anomalies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Aggregation functions for real-time data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Mean,
    Median,
    Mode,
    Min,
    Max,
    Sum,
    Count,
    StandardDeviation,
    Variance,
    Percentile(f64),
    Rate,
    Custom(String),
}
/// Enhanced intelligent buffering system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligentBufferingConfig {
    /// Enable intelligent buffering
    pub enable_intelligent_buffering: bool,
    /// Buffer size adjustment strategy
    pub buffer_strategy: BufferStrategy,
    /// Minimum buffer size
    pub min_buffer_size: usize,
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Buffer utilization threshold for adjustment
    pub utilization_threshold: f32,
    /// Buffer adjustment factor
    pub adjustment_factor: f32,
    /// Enable priority-based buffering
    pub enable_priority_buffering: bool,
    /// Memory pressure threshold for buffer reduction
    pub memory_pressure_threshold: f32,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventPriority {
    Critical,
    High,
    Medium,
    Low,
    Background,
}
/// Event buffer for stream data retention
#[derive(Debug)]
struct EventBuffer {
    events: Vec<(SystemTime, StreamEvent)>,
    max_retention: Duration,
    max_size: usize,
}
impl EventBuffer {
    fn new(max_retention_seconds: u64, max_size: usize) -> Self {
        Self {
            events: Vec::new(),
            max_retention: Duration::from_secs(max_retention_seconds),
            max_size,
        }
    }
    fn add_event(&mut self, event: StreamEvent) {
        let now = SystemTime::now();
        self.events.push((now, event));
        self.cleanup_old_events();
        if self.events.len() > self.max_size {
            self.events.drain(0..self.events.len() - self.max_size);
        }
    }
    fn cleanup_old_events(&mut self) {
        let cutoff = SystemTime::now()
            .checked_sub(self.max_retention)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.events.retain(|(timestamp, _)| timestamp >= &cutoff);
    }
    fn get_events_since(&self, since: SystemTime) -> Vec<StreamEvent> {
        self.events
            .iter()
            .filter(|(timestamp, _)| timestamp >= &since)
            .map(|(_, event)| event.clone())
            .collect()
    }
}
/// Real-time streaming debugger
#[derive(Debug)]
pub struct StreamingDebugger {
    config: StreamingDebugConfig,
    event_sender: broadcast::Sender<StreamEvent>,
    subscribers: Arc<RwLock<HashMap<Uuid, StreamSubscriber>>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    event_buffer: Arc<RwLock<EventBuffer>>,
    stream_metrics: Arc<RwLock<StreamMetrics>>,
    pub(super) is_running: Arc<RwLock<bool>>,
}
impl StreamingDebugger {
    /// Create a new streaming debugger
    pub fn new(config: StreamingDebugConfig) -> Self {
        let (event_sender, _) = broadcast::channel(config.stream_buffer_size);
        Self {
            event_sender,
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(config.rate_limit_per_second))),
            event_buffer: Arc::new(RwLock::new(EventBuffer::new(
                config.max_retention_seconds,
                config.stream_buffer_size * 2,
            ))),
            stream_metrics: Arc::new(RwLock::new(StreamMetrics::default())),
            is_running: Arc::new(RwLock::new(false)),
            config,
        }
    }
    /// Start the streaming debugger
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Ok(());
        }
        info!(
            "Starting streaming debugger with {} buffer size",
            self.config.stream_buffer_size
        );
        *is_running = true;
        self.start_background_tasks().await?;
        let control_event = StreamEvent::StreamControl {
            event_type: StreamControlType::StreamStarted,
            stream_id: Uuid::new_v4(),
            message: "Streaming debugger started".to_string(),
            timestamp: SystemTime::now(),
        };
        self.send_event(control_event).await?;
        Ok(())
    }
    /// Stop the streaming debugger
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            return Ok(());
        }
        info!("Stopping streaming debugger");
        *is_running = false;
        let control_event = StreamEvent::StreamControl {
            event_type: StreamControlType::StreamStopped,
            stream_id: Uuid::new_v4(),
            message: "Streaming debugger stopped".to_string(),
            timestamp: SystemTime::now(),
        };
        self.send_event(control_event).await?;
        Ok(())
    }
    /// Subscribe to stream events
    pub async fn subscribe(
        &self,
        name: String,
        format: StreamFormat,
        filter: StreamFilter,
    ) -> Result<StreamSubscription> {
        let subscriber_id = Uuid::new_v4();
        let subscriber = StreamSubscriber {
            id: subscriber_id,
            name: name.clone(),
            format,
            filter: filter.clone(),
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
        };
        let mut subscribers = self.subscribers.write().await;
        subscribers.insert(subscriber_id, subscriber);
        info!("New stream subscriber: {} ({})", name, subscriber_id);
        {
            let mut metrics = self.stream_metrics.write().await;
            metrics.active_subscribers = subscribers.len();
        }
        let receiver = self.event_sender.subscribe();
        let subscription = StreamSubscription::new(subscriber_id, receiver, filter, format);
        let control_event = StreamEvent::StreamControl {
            event_type: StreamControlType::ConnectionEstablished,
            stream_id: subscriber_id,
            message: format!("Subscriber {} connected", name),
            timestamp: SystemTime::now(),
        };
        self.send_event(control_event).await?;
        Ok(subscription)
    }
    /// Unsubscribe from stream events
    pub async fn unsubscribe(&self, subscriber_id: Uuid) -> Result<()> {
        let mut subscribers = self.subscribers.write().await;
        if let Some(subscriber) = subscribers.remove(&subscriber_id) {
            info!(
                "Removed stream subscriber: {} ({})",
                subscriber.name, subscriber_id
            );
            let mut metrics = self.stream_metrics.write().await;
            metrics.active_subscribers = subscribers.len();
            let control_event = StreamEvent::StreamControl {
                event_type: StreamControlType::ConnectionLost,
                stream_id: subscriber_id,
                message: format!("Subscriber {} disconnected", subscriber.name),
                timestamp: SystemTime::now(),
            };
            drop(subscribers);
            self.send_event(control_event).await?;
        }
        Ok(())
    }
    /// Send a stream event
    pub async fn send_event(&self, event: StreamEvent) -> Result<()> {
        if !*self.is_running.read().await {
            return Ok(());
        }
        {
            let mut rate_limiter = self.rate_limiter.write().await;
            if !rate_limiter.try_consume() {
                let mut metrics = self.stream_metrics.write().await;
                metrics.rate_limited_events += 1;
                return Ok(());
            }
        }
        {
            let mut buffer = self.event_buffer.write().await;
            buffer.add_event(event.clone());
        }
        match self.event_sender.send(event) {
            Ok(subscriber_count) => {
                let mut metrics = self.stream_metrics.write().await;
                metrics.total_events_sent += 1;
                metrics.last_update = SystemTime::now();
                debug!("Sent event to {} subscribers", subscriber_count);
            },
            Err(broadcast::error::SendError(_)) => {
                warn!("No active subscribers for stream event");
            },
        }
        Ok(())
    }
    /// Send tensor data event
    pub async fn send_tensor_data(
        &self,
        session_id: Uuid,
        tensor_id: Uuid,
        name: String,
        shape: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<()> {
        let statistics = self.compute_tensor_statistics(&values);
        let event = StreamEvent::TensorData {
            session_id,
            tensor_id,
            name,
            shape,
            values,
            statistics,
            timestamp: SystemTime::now(),
        };
        self.send_event(event).await
    }
    /// Send gradient flow event
    pub async fn send_gradient_flow(
        &self,
        session_id: Uuid,
        layer_name: String,
        gradients: &[f64],
    ) -> Result<()> {
        let gradient_statistics = self.compute_gradient_statistics(gradients);
        let gradient_norm = gradients.iter().map(|x| x * x).sum::<f64>().sqrt();
        let vanishing_risk = self.assess_vanishing_gradient_risk(gradients);
        let exploding_risk = self.assess_exploding_gradient_risk(gradients);
        let event = StreamEvent::GradientFlow {
            session_id,
            layer_name,
            gradient_norm,
            gradient_statistics,
            vanishing_risk,
            exploding_risk,
            timestamp: SystemTime::now(),
        };
        self.send_event(event).await
    }
    /// Send performance metrics event
    pub async fn send_performance_metrics(
        &self,
        session_id: Uuid,
        operation_name: String,
        latency_ms: f64,
        memory_usage_mb: f64,
        cpu_utilization: f64,
        gpu_utilization: Option<f64>,
    ) -> Result<()> {
        let event = StreamEvent::PerformanceMetrics {
            session_id,
            operation_name,
            latency_ms,
            memory_usage_mb,
            cpu_utilization,
            gpu_utilization,
            timestamp: SystemTime::now(),
        };
        self.send_event(event).await
    }
    /// Send training dynamics event
    pub async fn send_training_dynamics(
        &self,
        session_id: Uuid,
        epoch: u32,
        step: u64,
        loss: f64,
        learning_rate: f64,
        accuracy: Option<f64>,
        convergence_score: f64,
    ) -> Result<()> {
        let event = StreamEvent::TrainingDynamics {
            session_id,
            epoch,
            step,
            loss,
            learning_rate,
            accuracy,
            convergence_score,
            timestamp: SystemTime::now(),
        };
        self.send_event(event).await
    }
    /// Send anomaly detection event
    pub async fn send_anomaly_detected(
        &self,
        session_id: Uuid,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        description: String,
        confidence: f64,
        affected_components: Vec<String>,
    ) -> Result<()> {
        let event = StreamEvent::AnomalyDetected {
            session_id,
            anomaly_type,
            severity,
            description,
            confidence,
            affected_components,
            timestamp: SystemTime::now(),
        };
        self.send_event(event).await
    }
    /// Send system health event
    pub async fn send_system_health(
        &self,
        session_id: Uuid,
        health_score: f64,
        memory_pressure: f64,
        resource_utilization: HashMap<String, f64>,
        active_warnings: Vec<String>,
    ) -> Result<()> {
        let event = StreamEvent::SystemHealth {
            session_id,
            health_score,
            memory_pressure,
            resource_utilization,
            active_warnings,
            timestamp: SystemTime::now(),
        };
        self.send_event(event).await
    }
    /// Get current stream metrics
    pub async fn get_metrics(&self) -> StreamMetrics {
        let metrics = self.stream_metrics.read().await;
        metrics.clone()
    }
    /// Get events since a specific timestamp
    pub async fn get_events_since(&self, since: SystemTime) -> Vec<StreamEvent> {
        let buffer = self.event_buffer.read().await;
        buffer.get_events_since(since)
    }
    /// Get list of active subscribers
    pub async fn get_subscribers(&self) -> Vec<StreamSubscriber> {
        let subscribers = self.subscribers.read().await;
        subscribers.values().cloned().collect()
    }
    /// Start background tasks for maintenance and metrics
    async fn start_background_tasks(&self) -> Result<()> {
        let metrics = self.stream_metrics.clone();
        let is_running = self.is_running.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            let mut last_event_count = 0u64;
            loop {
                interval.tick().await;
                if !*is_running.read().await {
                    break;
                }
                let mut metrics_guard = metrics.write().await;
                let current_events = metrics_guard.total_events_sent;
                metrics_guard.events_per_second = (current_events - last_event_count) as f64;
                last_event_count = current_events;
                debug!(
                    "Stream metrics: {} events/sec, {} active subscribers",
                    metrics_guard.events_per_second, metrics_guard.active_subscribers
                );
            }
        });
        Ok(())
    }
    /// Compute tensor statistics
    pub(super) fn compute_tensor_statistics(&self, values: &[f64]) -> TensorStatistics {
        if values.is_empty() {
            return TensorStatistics {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                sparsity: 0.0,
            };
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let nan_count = values.iter().filter(|x| x.is_nan()).count();
        let inf_count = values.iter().filter(|x| x.is_infinite()).count();
        let zero_count = values.iter().filter(|&&x| x == 0.0).count();
        let sparsity = zero_count as f64 / values.len() as f64;
        TensorStatistics {
            mean,
            std,
            min,
            max,
            nan_count,
            inf_count,
            zero_count,
            sparsity,
        }
    }
    /// Compute gradient statistics
    pub(super) fn compute_gradient_statistics(&self, gradients: &[f64]) -> GradientStatistics {
        if gradients.is_empty() {
            return GradientStatistics {
                l1_norm: 0.0,
                l2_norm: 0.0,
                max_grad: 0.0,
                min_grad: 0.0,
                dead_neuron_ratio: 0.0,
                gradient_diversity: 0.0,
            };
        }
        let l1_norm = gradients.iter().map(|x| x.abs()).sum::<f64>();
        let l2_norm = gradients.iter().map(|x| x * x).sum::<f64>().sqrt();
        let max_grad = gradients.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_grad = gradients.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let dead_threshold = 1e-7;
        let dead_count = gradients.iter().filter(|&&x| x.abs() < dead_threshold).count();
        let dead_neuron_ratio = dead_count as f64 / gradients.len() as f64;
        let mean = gradients.iter().sum::<f64>() / gradients.len() as f64;
        let variance =
            gradients.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / gradients.len() as f64;
        let std = variance.sqrt();
        let gradient_diversity = if mean.abs() > 1e-10 { std / mean.abs() } else { 0.0 };
        GradientStatistics {
            l1_norm,
            l2_norm,
            max_grad,
            min_grad,
            dead_neuron_ratio,
            gradient_diversity,
        }
    }
    /// Assess vanishing gradient risk
    fn assess_vanishing_gradient_risk(&self, gradients: &[f64]) -> f64 {
        if gradients.is_empty() {
            return 0.0;
        }
        let l2_norm = gradients.iter().map(|x| x * x).sum::<f64>().sqrt();
        let avg_grad_magnitude = l2_norm / gradients.len() as f64;
        let vanishing_threshold = 1e-5;
        if avg_grad_magnitude < vanishing_threshold {
            1.0 - (avg_grad_magnitude / vanishing_threshold)
        } else {
            0.0
        }
    }
    /// Assess exploding gradient risk
    fn assess_exploding_gradient_risk(&self, gradients: &[f64]) -> f64 {
        if gradients.is_empty() {
            return 0.0;
        }
        let l2_norm = gradients.iter().map(|x| x * x).sum::<f64>().sqrt();
        let avg_grad_magnitude = l2_norm / gradients.len() as f64;
        let exploding_threshold = 10.0;
        if avg_grad_magnitude > exploding_threshold {
            (avg_grad_magnitude / exploding_threshold - 1.0).min(1.0)
        } else {
            0.0
        }
    }
    /// Alias for start method to match enhanced streaming debugger interface
    pub async fn start_streaming(&self) -> Result<()> {
        self.start().await
    }
    /// Alias for stop method to match enhanced streaming debugger interface
    pub async fn stop_streaming(&self) -> Result<()> {
        self.stop().await
    }
    /// Stream a generic event
    pub async fn stream_event(&self, event: StreamEvent) -> Result<()> {
        self.send_event(event).await
    }
}
/// Metrics for streaming performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    pub total_events_sent: u64,
    pub events_per_second: f64,
    pub active_subscribers: usize,
    pub dropped_events: u64,
    pub rate_limited_events: u64,
    pub average_latency_ms: f64,
    pub buffer_utilization: f64,
    pub error_count: u64,
    pub last_update: SystemTime,
}
/// Filtered event stream implementation
pub struct FilteredEventStream {
    pub(super) receiver: BroadcastStream<StreamEvent>,
    pub(super) filter: StreamFilter,
}
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub recency_weight: f32,
    pub severity_weight: f32,
    pub frequency_weight: f32,
    pub user_attention_weight: f32,
    pub system_impact_weight: f32,
}
/// Stream subscriber information
#[derive(Debug, Clone)]
pub struct StreamSubscriber {
    pub id: Uuid,
    pub name: String,
    pub format: StreamFormat,
    pub filter: StreamFilter,
    pub created_at: SystemTime,
    pub last_activity: SystemTime,
}
/// Network condition monitoring
#[derive(Debug)]
pub struct NetworkConditionMonitor {
    /// Current bandwidth (bytes/sec)
    pub current_bandwidth: u64,
    /// Current latency (ms)
    pub current_latency: u64,
    /// Packet loss percentage
    pub packet_loss: f32,
    /// Connection quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Historical measurements
    pub history: Vec<NetworkMeasurement>,
    /// Measurement timestamps
    pub last_measurement: Instant,
}
impl NetworkConditionMonitor {
    pub fn new() -> Self {
        Self {
            current_bandwidth: 1_000_000,
            current_latency: 50,
            packet_loss: 0.0,
            quality_score: 1.0,
            history: Vec::new(),
            last_measurement: Instant::now(),
        }
    }
    pub async fn update_conditions(&mut self) {
        let mut rng = thread_rng();
        self.current_bandwidth = 800_000 + (rng.random::<u32>() % 400_000) as u64;
        self.current_latency = 30 + (rng.random::<u32>() % 100) as u64;
        self.packet_loss = (rng.random::<f32>() * 0.05).max(0.0);
        self.quality_score = self.calculate_quality_score();
        let measurement = NetworkMeasurement {
            timestamp: Instant::now(),
            bandwidth: self.current_bandwidth,
            latency: self.current_latency,
            packet_loss: self.packet_loss,
        };
        self.history.push(measurement);
        if self.history.len() > 100 {
            self.history.remove(0);
        }
        self.last_measurement = Instant::now();
    }
    fn calculate_quality_score(&self) -> f32 {
        let bandwidth_score = (self.current_bandwidth as f32 / 1_000_000.0).min(1.0);
        let latency_score = (200.0 - self.current_latency as f32).max(0.0) / 200.0;
        let packet_loss_score = (1.0 - self.packet_loss).max(0.0);
        (bandwidth_score * 0.4 + latency_score * 0.4 + packet_loss_score * 0.2)
            .max(0.0)
            .min(1.0)
    }
}
