//! Real-Time Analysis System
//!
//! Continuous analysis of performance data to support optimization decision-making

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{task::JoinHandle, time::interval};
use tracing::{info, warn};

use super::super::types::*;
use super::support::AnalysisResult;

// =============================================================================

/// Real-time analyzer for optimization engine
///
/// Provides continuous analysis of performance data to support
/// optimization decision-making with multiple analysis algorithms.
pub struct RealTimeAnalyzer {
    /// Analysis algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn RealTimeAnalysisAlgorithm + Send + Sync>>>>,

    /// Analysis results cache
    results: Arc<RwLock<HashMap<String, AnalysisResult>>>,

    /// Analysis statistics
    stats: Arc<RealTimeAnalysisStats>,

    /// Configuration
    config: Arc<RwLock<RealTimeAnalysisConfig>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Background analysis task
    background_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

/// Statistics for real-time analyzer
#[derive(Debug, Default)]
pub struct RealTimeAnalysisStats {
    /// Analyses performed
    pub analyses_performed: AtomicU64,

    /// Average analysis time
    pub avg_analysis_time_ms: AtomicF32,

    /// Cache hit rate
    pub cache_hit_rate: AtomicF32,

    /// Active algorithms
    pub active_algorithms: AtomicUsize,
}

/// Configuration for real-time analyzer
#[derive(Debug, Clone)]
pub struct RealTimeAnalysisConfig {
    /// Analysis interval
    pub analysis_interval: Duration,

    /// Cache TTL
    pub cache_ttl: Duration,

    /// Maximum cache size
    pub max_cache_size: usize,

    /// Analysis timeout
    pub analysis_timeout: Duration,
}

impl Default for RealTimeAnalysisConfig {
    fn default() -> Self {
        Self {
            analysis_interval: Duration::from_secs(30),
            cache_ttl: Duration::from_secs(300),
            max_cache_size: 1000,
            analysis_timeout: Duration::from_secs(15),
        }
    }
}

/// Trait for real-time analysis algorithms
pub trait RealTimeAnalysisAlgorithm: Send + Sync {
    /// Analyze real-time data
    fn analyze(
        &self,
        metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
    ) -> Result<AnalysisResult, RealTimeMetricsError>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Check if algorithm is applicable for current data
    fn is_applicable(&self, metrics: &RealTimeMetrics) -> bool;

    /// Get algorithm confidence in analysis
    fn confidence(&self, data_quality: f32) -> f32;
}

impl RealTimeAnalyzer {
    /// Create a new real-time analyzer
    pub async fn new() -> Result<Self> {
        let analyzer = Self {
            algorithms: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RealTimeAnalysisStats::default()),
            config: Arc::new(RwLock::new(RealTimeAnalysisConfig::default())),
            shutdown: Arc::new(AtomicBool::new(false)),
            background_task: Arc::new(Mutex::new(None)),
        };

        analyzer.initialize_algorithms().await?;
        Ok(analyzer)
    }

    /// Start real-time analyzer
    pub async fn start(&self) -> Result<()> {
        info!("Starting real-time analyzer");

        // Start background analysis task
        self.start_background_analysis().await?;

        Ok(())
    }

    /// Perform real-time analysis
    pub async fn analyze(
        &self,
        metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
    ) -> Result<HashMap<String, AnalysisResult>> {
        let start_time = Instant::now();
        let algorithms = self.algorithms.lock();
        let mut results: HashMap<String, AnalysisResult> = HashMap::new();

        for algorithm in algorithms.iter() {
            if algorithm.is_applicable(metrics) {
                match algorithm.analyze(metrics, history) {
                    Ok(result) => {
                        results.insert(algorithm.name().to_string(), result);
                    },
                    Err(e) => {
                        warn!("Analysis error from {}: {}", algorithm.name(), e);
                    },
                }
            }
        }

        // Update statistics
        self.stats.analyses_performed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .avg_analysis_time_ms
            .store(start_time.elapsed().as_millis() as f32, Ordering::Relaxed);

        // Cache results
        let mut cache = self.results.write();
        for (name, result) in results.iter() {
            cache.insert(name.clone(), result.clone());
        }

        Ok(results)
    }

    /// Get cached analysis results
    pub async fn get_cached_results(&self) -> HashMap<String, AnalysisResult> {
        let results = self.results.read();
        results.clone()
    }

    /// Shutdown real-time analyzer
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);

        // Stop background task
        if let Some(task) = self.background_task.lock().take() {
            task.abort();
        }

        info!("Real-time analyzer shutdown complete");
        Ok(())
    }

    async fn initialize_algorithms(&self) -> Result<()> {
        let mut algorithms = self.algorithms.lock();

        algorithms.push(Box::new(PerformanceAnalysisAlgorithm::new()));
        algorithms.push(Box::new(TrendAnalysisAlgorithm::new()));
        algorithms.push(Box::new(BottleneckAnalysisAlgorithm::new()));
        algorithms.push(Box::new(PredictiveAnalysisAlgorithm::new()));

        self.stats.active_algorithms.store(algorithms.len(), Ordering::Relaxed);

        info!(
            "Initialized {} real-time analysis algorithms",
            algorithms.len()
        );
        Ok(())
    }

    async fn start_background_analysis(&self) -> Result<()> {
        let analyzer_clone = self.clone();
        let task = tokio::spawn(async move {
            let analysis_interval = {
                let config = analyzer_clone.config.read();
                config.analysis_interval
            };
            let mut interval = interval(analysis_interval);

            while !analyzer_clone.shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                // Perform cache cleanup
                analyzer_clone.cleanup_cache().await;
            }
        });

        *self.background_task.lock() = Some(task);
        Ok(())
    }

    async fn cleanup_cache(&self) {
        let config = self.config.read();
        let ttl = config.cache_ttl;
        let max_size = config.max_cache_size;
        drop(config);

        let mut cache = self.results.write();
        let cutoff_time = Utc::now() - chrono::Duration::from_std(ttl).unwrap_or_default();

        // Remove expired entries
        cache.retain(|_, result| result.timestamp > cutoff_time);

        // Limit cache size
        if cache.len() > max_size {
            let excess = cache.len() - max_size;
            let keys_to_remove: Vec<String> = cache.keys().take(excess).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }
}

// Clone implementation for RealTimeAnalyzer
impl Clone for RealTimeAnalyzer {
    fn clone(&self) -> Self {
        Self {
            algorithms: Arc::clone(&self.algorithms),
            results: Arc::clone(&self.results),
            stats: Arc::clone(&self.stats),
            config: Arc::clone(&self.config),
            shutdown: Arc::clone(&self.shutdown),
            background_task: Arc::clone(&self.background_task),
        }
    }
}

/// Performance analysis algorithm
///
/// Analyzes current performance metrics and identifies optimization opportunities.
pub struct PerformanceAnalysisAlgorithm {
    analysis_history: VecDeque<PerformanceAnalysisRecord>,
}

#[derive(Debug, Clone)]
struct PerformanceAnalysisRecord {
    timestamp: DateTime<Utc>,
    cpu_efficiency: f32,
    memory_efficiency: f32,
    throughput_score: f32,
    latency_score: f32,
}

impl Default for PerformanceAnalysisAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceAnalysisAlgorithm {
    pub fn new() -> Self {
        Self {
            analysis_history: VecDeque::new(),
        }
    }

    fn calculate_performance_scores(&self, metrics: &RealTimeMetrics) -> (f32, f32, f32, f32) {
        // CPU efficiency (inverse of utilization for efficiency)
        let cpu_efficiency = if metrics.current_cpu_utilization > 0.0 {
            1.0 - (metrics.current_cpu_utilization - 0.7).max(0.0) / 0.3
        } else {
            0.5
        };

        // Memory efficiency
        let memory_efficiency = if metrics.current_memory_utilization > 0.0 {
            1.0 - (metrics.current_memory_utilization - 0.7).max(0.0) / 0.3
        } else {
            0.5
        };

        // Throughput score (normalized)
        let throughput_score = ((metrics.current_throughput / 100.0).min(1.0)) as f32;

        // Latency score (inverse of latency)
        let latency_ms = metrics.current_latency.as_millis() as f32;
        let latency_score =
            if latency_ms > 0.0 { (1000.0 / (1000.0 + latency_ms)).min(1.0) } else { 1.0 };

        (
            cpu_efficiency,
            memory_efficiency,
            throughput_score,
            latency_score,
        )
    }
}

impl RealTimeAnalysisAlgorithm for PerformanceAnalysisAlgorithm {
    fn analyze(
        &self,
        metrics: &RealTimeMetrics,
        _history: &[TimestampedMetrics],
    ) -> Result<AnalysisResult, RealTimeMetricsError> {
        let (cpu_efficiency, memory_efficiency, throughput_score, latency_score) =
            self.calculate_performance_scores(metrics);

        let overall_score =
            (cpu_efficiency + memory_efficiency + throughput_score + latency_score) / 4.0;

        let insights = vec![
            format!("CPU efficiency: {:.1}%", cpu_efficiency * 100.0),
            format!("Memory efficiency: {:.1}%", memory_efficiency * 100.0),
            format!("Throughput score: {:.1}%", throughput_score * 100.0),
            format!("Latency score: {:.1}%", latency_score * 100.0),
        ];

        let recommendations = if overall_score < 0.7 {
            vec!["Consider performance optimization".to_string()]
        } else {
            vec!["Performance is within acceptable range".to_string()]
        };

        Ok(AnalysisResult {
            algorithm_name: "performance_analysis".to_string(),
            timestamp: Utc::now(),
            confidence: overall_score,
            insights,
            recommendations,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("overall_score".to_string(), overall_score.to_string());
                metadata.insert("cpu_efficiency".to_string(), cpu_efficiency.to_string());
                metadata.insert(
                    "memory_efficiency".to_string(),
                    memory_efficiency.to_string(),
                );
                metadata.insert("throughput_score".to_string(), throughput_score.to_string());
                metadata.insert("latency_score".to_string(), latency_score.to_string());
                metadata
            },
        })
    }

    fn name(&self) -> &str {
        "performance_analysis"
    }

    fn is_applicable(&self, _metrics: &RealTimeMetrics) -> bool {
        true
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality * 0.9
    }
}

/// Trend analysis algorithm
///
/// Analyzes performance trends over time to identify patterns and predict future behavior.
pub struct TrendAnalysisAlgorithm {
    trend_history: VecDeque<TrendDataPoint>,
}

#[derive(Debug, Clone)]
struct TrendDataPoint {
    timestamp: DateTime<Utc>,
    cpu_trend: f32,
    memory_trend: f32,
    throughput_trend: f32,
    latency_trend: f32,
}

impl Default for TrendAnalysisAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalysisAlgorithm {
    pub fn new() -> Self {
        Self {
            trend_history: VecDeque::new(),
        }
    }

    fn calculate_trends(&self, history: &[TimestampedMetrics]) -> (f32, f32, f32, f32) {
        if history.len() < 2 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let recent = &history[history.len() - 1];
        let older = &history[0];

        let cpu_trend =
            recent.metrics.current_cpu_utilization - older.metrics.current_cpu_utilization;
        let memory_trend =
            recent.metrics.current_memory_utilization - older.metrics.current_memory_utilization;
        let throughput_trend =
            (recent.metrics.current_throughput - older.metrics.current_throughput) as f32;
        let latency_trend = (recent.metrics.current_latency.as_millis() as i64
            - older.metrics.current_latency.as_millis() as i64) as f32;

        (cpu_trend, memory_trend, throughput_trend, latency_trend)
    }
}

impl RealTimeAnalysisAlgorithm for TrendAnalysisAlgorithm {
    fn analyze(
        &self,
        _metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
    ) -> Result<AnalysisResult, RealTimeMetricsError> {
        let (cpu_trend, memory_trend, throughput_trend, latency_trend) =
            self.calculate_trends(history);

        let insights = vec![
            format!("CPU utilization trend: {:.3}", cpu_trend),
            format!("Memory utilization trend: {:.3}", memory_trend),
            format!("Throughput trend: {:.1}", throughput_trend),
            format!("Latency trend: {:.1}ms", latency_trend),
        ];

        let mut recommendations = Vec::new();

        if cpu_trend > 0.1 {
            recommendations
                .push("CPU utilization is increasing - consider optimization".to_string());
        }
        if memory_trend > 0.1 {
            recommendations
                .push("Memory utilization is increasing - monitor for leaks".to_string());
        }
        if throughput_trend < -5.0 {
            recommendations.push("Throughput is decreasing - investigate bottlenecks".to_string());
        }
        if latency_trend > 100.0 {
            recommendations.push("Latency is increasing - optimize response times".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Trends are stable".to_string());
        }

        let confidence = if history.len() >= 10 { 0.9 } else { 0.7 };

        Ok(AnalysisResult {
            algorithm_name: "trend_analysis".to_string(),
            timestamp: Utc::now(),
            confidence,
            insights,
            recommendations,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("cpu_trend".to_string(), cpu_trend.to_string());
                metadata.insert("memory_trend".to_string(), memory_trend.to_string());
                metadata.insert("throughput_trend".to_string(), throughput_trend.to_string());
                metadata.insert("latency_trend".to_string(), latency_trend.to_string());
                metadata.insert("data_points".to_string(), history.len().to_string());
                metadata
            },
        })
    }

    fn name(&self) -> &str {
        "trend_analysis"
    }

    fn is_applicable(&self, _metrics: &RealTimeMetrics) -> bool {
        true
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality * 0.85
    }
}

/// Bottleneck analysis algorithm
///
/// Identifies performance bottlenecks and resource constraints in the system.
pub struct BottleneckAnalysisAlgorithm {
    bottleneck_history: VecDeque<BottleneckRecord>,
}

#[derive(Debug, Clone)]
struct BottleneckRecord {
    timestamp: DateTime<Utc>,
    bottleneck_type: BottleneckType,
    severity: f32,
    resolution_suggested: String,
}

#[derive(Debug, Clone)]
enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Concurrency,
    Unknown,
}

impl Default for BottleneckAnalysisAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl BottleneckAnalysisAlgorithm {
    pub fn new() -> Self {
        Self {
            bottleneck_history: VecDeque::new(),
        }
    }

    fn identify_bottlenecks(&self, metrics: &RealTimeMetrics) -> Vec<(BottleneckType, f32)> {
        let mut bottlenecks = Vec::new();

        // CPU bottleneck
        if metrics.current_cpu_utilization > 0.9 {
            bottlenecks.push((BottleneckType::CPU, metrics.current_cpu_utilization));
        }

        // Memory bottleneck
        if metrics.current_memory_utilization > 0.85 {
            bottlenecks.push((BottleneckType::Memory, metrics.current_memory_utilization));
        }

        // I/O bottleneck (inferred from high latency)
        if metrics.current_latency.as_millis() > 1000 {
            let severity = (metrics.current_latency.as_millis() as f32 / 2000.0).min(1.0);
            bottlenecks.push((BottleneckType::IO, severity));
        }

        // Network bottleneck (inferred from low throughput with high latency)
        if metrics.current_throughput < 30.0 && metrics.current_latency.as_millis() > 500 {
            let severity = (500.0 / metrics.current_latency.as_millis() as f32).min(1.0);
            bottlenecks.push((BottleneckType::Network, severity));
        }

        bottlenecks
    }

    fn generate_bottleneck_recommendations(
        &self,
        bottlenecks: &[(BottleneckType, f32)],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        for (bottleneck_type, severity) in bottlenecks {
            match bottleneck_type {
                BottleneckType::CPU => {
                    recommendations.push(format!(
                        "CPU bottleneck detected (severity: {:.1}%) - Consider reducing CPU-intensive operations or scaling horizontally",
                        severity * 100.0
                    ));
                },
                BottleneckType::Memory => {
                    recommendations.push(format!(
                        "Memory bottleneck detected (severity: {:.1}%) - Optimize memory usage or increase available memory",
                        severity * 100.0
                    ));
                },
                BottleneckType::IO => {
                    recommendations.push(format!(
                        "I/O bottleneck detected (severity: {:.1}%) - Optimize disk access patterns or use faster storage",
                        severity * 100.0
                    ));
                },
                BottleneckType::Network => {
                    recommendations.push(format!(
                        "Network bottleneck detected (severity: {:.1}%) - Optimize network usage or improve bandwidth",
                        severity * 100.0
                    ));
                },
                BottleneckType::Concurrency => {
                    recommendations.push(format!(
                        "Concurrency bottleneck detected (severity: {:.1}%) - Optimize thread usage or reduce contention",
                        severity * 100.0
                    ));
                },
                BottleneckType::Unknown => {
                    recommendations.push(
                        "Unknown bottleneck detected - perform detailed profiling".to_string(),
                    );
                },
            }
        }

        if recommendations.is_empty() {
            recommendations.push("No significant bottlenecks detected".to_string());
        }

        recommendations
    }
}

impl RealTimeAnalysisAlgorithm for BottleneckAnalysisAlgorithm {
    fn analyze(
        &self,
        metrics: &RealTimeMetrics,
        _history: &[TimestampedMetrics],
    ) -> Result<AnalysisResult, RealTimeMetricsError> {
        let bottlenecks = self.identify_bottlenecks(metrics);
        let recommendations = self.generate_bottleneck_recommendations(&bottlenecks);

        let insights: Vec<String> = bottlenecks
            .iter()
            .map(|(bt, severity)| format!("{:?} bottleneck: {:.1}%", bt, severity * 100.0))
            .collect();

        let confidence = if bottlenecks.is_empty() { 0.8 } else { 0.9 };

        Ok(AnalysisResult {
            algorithm_name: "bottleneck_analysis".to_string(),
            timestamp: Utc::now(),
            confidence,
            insights,
            recommendations,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert(
                    "bottleneck_count".to_string(),
                    bottlenecks.len().to_string(),
                );

                for (i, (bt, severity)) in bottlenecks.iter().enumerate() {
                    metadata.insert(format!("bottleneck_{}_type", i), format!("{:?}", bt));
                    metadata.insert(format!("bottleneck_{}_severity", i), severity.to_string());
                }

                metadata
            },
        })
    }

    fn name(&self) -> &str {
        "bottleneck_analysis"
    }

    fn is_applicable(&self, _metrics: &RealTimeMetrics) -> bool {
        true
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality * 0.88
    }
}

/// Predictive analysis algorithm
///
/// Uses historical data to predict future performance trends and potential issues.
pub struct PredictiveAnalysisAlgorithm {
    prediction_history: VecDeque<PredictionRecord>,
    model_coefficients: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
struct PredictionRecord {
    timestamp: DateTime<Utc>,
    predicted_cpu: f32,
    predicted_memory: f32,
    predicted_throughput: f64,
    predicted_latency: Duration,
    confidence: f32,
}

impl Default for PredictiveAnalysisAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictiveAnalysisAlgorithm {
    pub fn new() -> Self {
        let mut coefficients = HashMap::new();

        // Simple linear model coefficients (would be trained in a real implementation)
        coefficients.insert("cpu_trend".to_string(), 0.7);
        coefficients.insert("memory_trend".to_string(), 0.8);
        coefficients.insert("throughput_trend".to_string(), 0.6);
        coefficients.insert("latency_trend".to_string(), 0.9);

        Self {
            prediction_history: VecDeque::new(),
            model_coefficients: coefficients,
        }
    }

    fn predict_future_metrics(&self, history: &[TimestampedMetrics]) -> Option<PredictionRecord> {
        if history.len() < 3 {
            return None;
        }

        // Simple trend-based prediction
        let recent_points = &history[history.len().saturating_sub(3)..];

        let cpu_trend = self.calculate_trend(
            &recent_points
                .iter()
                .map(|p| p.metrics.current_cpu_utilization)
                .collect::<Vec<_>>(),
        );

        let memory_trend = self.calculate_trend(
            &recent_points
                .iter()
                .map(|p| p.metrics.current_memory_utilization)
                .collect::<Vec<_>>(),
        );

        let throughput_trend = self.calculate_trend(
            &recent_points
                .iter()
                .map(|p| p.metrics.current_throughput as f32)
                .collect::<Vec<_>>(),
        );

        let latency_trend = self.calculate_trend(
            &recent_points
                .iter()
                .map(|p| p.metrics.current_latency.as_millis() as f32)
                .collect::<Vec<_>>(),
        );

        let current = &recent_points[recent_points.len() - 1].metrics;

        // Predict next values
        let predicted_cpu = (current.current_cpu_utilization + cpu_trend).clamp(0.0, 1.0);
        let predicted_memory = (current.current_memory_utilization + memory_trend).clamp(0.0, 1.0);
        let predicted_throughput =
            (current.current_throughput as f32 + throughput_trend).max(0.0) as f64;
        let predicted_latency = Duration::from_millis(
            (current.current_latency.as_millis() as f32 + latency_trend).max(0.0) as u64,
        );

        let confidence = self.calculate_prediction_confidence(history);

        Some(PredictionRecord {
            timestamp: Utc::now(),
            predicted_cpu,
            predicted_memory,
            predicted_throughput,
            predicted_latency,
            confidence,
        })
    }

    fn calculate_trend(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f32;
        let sum_x: f32 = (0..values.len()).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();

        // Linear regression slope
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        slope
    }

    fn calculate_prediction_confidence(&self, history: &[TimestampedMetrics]) -> f32 {
        let base_confidence = if history.len() >= 10 { 0.8 } else { 0.6 };

        // Adjust based on data stability
        let stability = self.calculate_data_stability(history);

        base_confidence * stability
    }

    fn calculate_data_stability(&self, history: &[TimestampedMetrics]) -> f32 {
        if history.len() < 3 {
            return 0.5;
        }

        let recent_points = &history[history.len().saturating_sub(5)..];
        let cpu_values: Vec<f32> =
            recent_points.iter().map(|p| p.metrics.current_cpu_utilization).collect();
        let memory_values: Vec<f32> =
            recent_points.iter().map(|p| p.metrics.current_memory_utilization).collect();

        let cpu_variance = self.calculate_variance(&cpu_values);
        let memory_variance = self.calculate_variance(&memory_values);

        // Lower variance indicates higher stability
        let stability = 1.0 - ((cpu_variance + memory_variance) / 2.0).min(1.0);
        stability.max(0.3) // Minimum stability
    }

    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }
}

impl RealTimeAnalysisAlgorithm for PredictiveAnalysisAlgorithm {
    fn analyze(
        &self,
        _metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
    ) -> Result<AnalysisResult, RealTimeMetricsError> {
        let prediction = self.predict_future_metrics(history);

        let (insights, recommendations, confidence, metadata) = if let Some(pred) = prediction {
            let insights = vec![
                format!(
                    "Predicted CPU utilization: {:.1}%",
                    pred.predicted_cpu * 100.0
                ),
                format!(
                    "Predicted memory utilization: {:.1}%",
                    pred.predicted_memory * 100.0
                ),
                format!("Predicted throughput: {:.1}", pred.predicted_throughput),
                format!(
                    "Predicted latency: {}ms",
                    pred.predicted_latency.as_millis()
                ),
            ];

            let mut recommendations = Vec::new();

            if pred.predicted_cpu > 0.9 {
                recommendations
                    .push("CPU utilization predicted to exceed 90% - prepare scaling".to_string());
            }
            if pred.predicted_memory > 0.85 {
                recommendations.push(
                    "Memory utilization predicted to exceed 85% - monitor for pressure".to_string(),
                );
            }
            if pred.predicted_latency.as_millis() > 1000 {
                recommendations
                    .push("Latency predicted to exceed 1s - investigate bottlenecks".to_string());
            }
            if pred.predicted_throughput < 20.0 {
                recommendations.push(
                    "Throughput predicted to drop below 20 - optimize performance".to_string(),
                );
            }

            if recommendations.is_empty() {
                recommendations.push("No concerning trends predicted".to_string());
            }

            let mut metadata = HashMap::new();
            metadata.insert("predicted_cpu".to_string(), pred.predicted_cpu.to_string());
            metadata.insert(
                "predicted_memory".to_string(),
                pred.predicted_memory.to_string(),
            );
            metadata.insert(
                "predicted_throughput".to_string(),
                pred.predicted_throughput.to_string(),
            );
            metadata.insert(
                "predicted_latency_ms".to_string(),
                pred.predicted_latency.as_millis().to_string(),
            );
            metadata.insert(
                "prediction_confidence".to_string(),
                pred.confidence.to_string(),
            );

            (insights, recommendations, pred.confidence, metadata)
        } else {
            (
                vec!["Insufficient data for prediction".to_string()],
                vec!["Collect more historical data for accurate predictions".to_string()],
                0.3,
                HashMap::new(),
            )
        };

        Ok(AnalysisResult {
            algorithm_name: "predictive_analysis".to_string(),
            timestamp: Utc::now(),
            confidence,
            insights,
            recommendations,
            metadata,
        })
    }

    fn name(&self) -> &str {
        "predictive_analysis"
    }

    fn is_applicable(&self, _metrics: &RealTimeMetrics) -> bool {
        true
    }

    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality * 0.7 // Predictive analysis inherently less certain
    }
}
