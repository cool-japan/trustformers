//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::profiling_pipeline::DataAggregationEngine;
use super::super::types::optimization::OptimizationPerformanceData;
use super::super::types::*;
use super::functions::DEFAULT_METRICS_BUFFER_SIZE;
// Split in 0.2.1: the anomaly/reporting/insights/switcher/processor engines
// live in `types_engines`, which in turn uses the collector and analyzer here.
use super::types_engines::{
    AdaptiveStrategySwitcher, AnomalyDetectionEngine, LiveInsightsGenerator, ProfileDataPoint,
    RealTimeReportingEngine, StreamingDataProcessor,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::task::JoinHandle;
use tokio::time::{interval, sleep};

/// Real-time performance counters for profiling statistics
#[derive(Debug)]
pub struct RealTimePerformanceCounters {
    data_points_processed: AtomicU64,
    anomalies_detected: AtomicU64,
    insights_generated: AtomicU64,
    processing_rate: AtomicU64,
}
impl RealTimePerformanceCounters {
    pub fn new() -> Self {
        Self {
            data_points_processed: AtomicU64::new(0),
            anomalies_detected: AtomicU64::new(0),
            insights_generated: AtomicU64::new(0),
            processing_rate: AtomicU64::new(0),
        }
    }
    pub fn increment_data_points_processed(&self) {
        self.data_points_processed.fetch_add(1, Ordering::Relaxed);
    }
    pub fn increment_anomalies_detected(&self) {
        self.anomalies_detected.fetch_add(1, Ordering::Relaxed);
    }
    pub fn increment_insights_generated(&self) {
        self.insights_generated.fetch_add(1, Ordering::Relaxed);
    }
    pub fn get_current_stats(&self) -> PerformanceCounterStats {
        PerformanceCounterStats {
            data_points_processed: self.data_points_processed.load(Ordering::Relaxed),
            anomalies_detected: self.anomalies_detected.load(Ordering::Relaxed),
            insights_generated: self.insights_generated.load(Ordering::Relaxed),
            processing_rate: self.processing_rate.load(Ordering::Relaxed),
        }
    }
}
/// Core real-time profiling engine providing continuous monitoring during test execution
///
/// The RealTimeTestProfiler is the central orchestrator for real-time profiling operations,
/// managing streaming analysis, adaptive optimization, and live insights generation with
/// minimal performance overhead. It provides comprehensive monitoring capabilities while
/// maintaining thread-safe concurrent operations.
#[derive(Debug)]
pub struct RealTimeTestProfiler {
    /// Profiler configuration
    config: Arc<RwLock<RealTimeProfilerConfig>>,
    /// Real-time metrics collector
    metrics_collector: Arc<RealTimeMetricsCollector>,
    /// Streaming analyzer
    streaming_analyzer: Arc<StreamingAnalyzer>,
    /// Adaptive optimizer
    adaptive_optimizer: Arc<AdaptiveOptimizer>,
    /// Streaming data processor
    data_processor: Arc<StreamingDataProcessor>,
    /// Anomaly detection engine
    anomaly_detector: Arc<AnomalyDetectionEngine>,
    /// Live insights generator
    insights_generator: Arc<LiveInsightsGenerator>,
    /// Performance trend analyzer
    trend_analyzer: Arc<PerformanceTrendAnalyzer>,
    /// Adaptive strategy switcher
    strategy_switcher: Arc<AdaptiveStrategySwitcher>,
    /// Real-time reporting engine
    reporting_engine: Arc<RealTimeReportingEngine>,
    /// Active profiling sessions
    active_sessions: Arc<Mutex<HashMap<String, ProfilingSession>>>,
    /// Real-time data stream
    data_stream: Arc<Mutex<VecDeque<ProfileDataPoint>>>,
    /// Profiling state
    profiling_active: Arc<AtomicBool>,
    /// Performance counters
    performance_counters: Arc<RealTimePerformanceCounters>,
    /// Control handles
    control_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl RealTimeTestProfiler {
    /// Create a new real-time test profiler with the specified configuration
    pub async fn new(config: RealTimeProfilerConfig) -> Result<Self> {
        let config_arc = Arc::new(RwLock::new(config.clone()));
        let metrics_collector_config = MetricsCollectorConfig::default();
        let metrics_collector =
            Arc::new(RealTimeMetricsCollector::new(metrics_collector_config).await?);
        let streaming_analyzer_config = StreamingAnalyzerConfig::default();
        let streaming_analyzer = Arc::new(StreamingAnalyzer::new(streaming_analyzer_config).await?);
        let adaptive_optimizer_config = AdaptiveOptimizerConfig::default();
        let adaptive_optimizer = Arc::new(AdaptiveOptimizer::new(adaptive_optimizer_config).await?);
        let data_processor_config = DataProcessorConfig::default();
        let data_processor = Arc::new(StreamingDataProcessor::new(data_processor_config).await?);
        let anomaly_detector_config = AnomalyDetectionConfig::default();
        let anomaly_detector =
            Arc::new(AnomalyDetectionEngine::new(anomaly_detector_config).await?);
        let insights_generator_config = InsightsGeneratorConfig::default();
        let insights_generator =
            Arc::new(LiveInsightsGenerator::new(insights_generator_config).await?);
        let trend_analyzer_config = TrendAnalyzerConfig::default();
        let trend_analyzer = Arc::new(PerformanceTrendAnalyzer::new(trend_analyzer_config).await?);
        let strategy_switcher_config = StrategySwitcherConfig::default();
        let strategy_switcher =
            Arc::new(AdaptiveStrategySwitcher::new(strategy_switcher_config).await?);
        let reporting_engine_config = ReportingEngineConfig::default();
        let reporting_engine =
            Arc::new(RealTimeReportingEngine::new(reporting_engine_config).await?);
        Ok(Self {
            config: config_arc,
            metrics_collector,
            streaming_analyzer,
            adaptive_optimizer,
            data_processor,
            anomaly_detector,
            insights_generator,
            trend_analyzer,
            strategy_switcher,
            reporting_engine,
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
            data_stream: Arc::new(Mutex::new(VecDeque::new())),
            profiling_active: Arc::new(AtomicBool::new(false)),
            performance_counters: Arc::new(RealTimePerformanceCounters::new()),
            control_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    fn cloned_config(&self) -> RealTimeProfilerConfig {
        let guard = self.config.read();
        guard.clone()
    }
    /// Start real-time profiling operations
    pub async fn start_profiling(&self) -> Result<()> {
        if self.profiling_active.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.profiling_active.store(true, Ordering::Relaxed);
        self.metrics_collector.start_collection().await?;
        self.streaming_analyzer.start_analysis().await?;
        self.adaptive_optimizer.start_optimization().await?;
        self.data_processor.start_processing().await?;
        self.anomaly_detector.start_detection().await?;
        self.insights_generator.start_generation().await?;
        self.trend_analyzer.start_analysis().await?;
        self.strategy_switcher.start_switching().await?;
        self.reporting_engine.start_reporting().await?;
        self.start_profiling_loop().await?;
        Ok(())
    }
    /// Stop real-time profiling operations
    pub async fn stop_profiling(&self) -> Result<()> {
        self.profiling_active.store(false, Ordering::Relaxed);
        // The guard must not survive into the `.await`s below: a
        // `parking_lot::MutexGuard` is not `Send`, which made the whole future
        // non-`Send` and would have blocked the executor thread while the
        // component shutdowns ran.
        {
            let mut handles = self.control_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.metrics_collector.stop_collection().await?;
        self.streaming_analyzer.stop_analysis().await?;
        self.adaptive_optimizer.stop_optimization().await?;
        self.data_processor.stop_processing().await?;
        self.anomaly_detector.stop_detection().await?;
        self.insights_generator.stop_generation().await?;
        self.trend_analyzer.stop_analysis().await?;
        self.strategy_switcher.stop_switching().await?;
        self.reporting_engine.stop_reporting().await?;
        Ok(())
    }
    /// Monitor test execution in real-time
    pub async fn monitor_test_execution(&self, test_id: &str) -> Result<()> {
        let session = ProfilingSession::new(test_id.to_string()).await?;
        self.active_sessions.lock().insert(test_id.to_string(), session);
        self.start_test_monitoring(test_id).await?;
        Ok(())
    }
    /// Get real-time insights and recommendations over the collected window.
    pub async fn get_live_insights(&self) -> Result<LiveInsights> {
        let samples = self.collected_metrics_snapshot();
        self.insights_generator.generate_current_insights(&samples).await
    }
    /// Snapshot the metrics carried by the profiler's data stream.
    ///
    /// Cloned out under a scoped lock so the non-`Send` guard cannot survive
    /// into an `.await` (see `RealTimeTestProfiler::stop_profiling`).
    fn collected_metrics_snapshot(&self) -> Vec<RealTimeMetrics> {
        let stream = self.data_stream.lock();
        stream.iter().map(|point| point.metrics.clone()).collect()
    }
    /// Generate comprehensive real-time performance report
    pub async fn generate_live_report(&self) -> Result<RealTimeReport> {
        self.reporting_engine.generate_comprehensive_report().await
    }
    /// Get current profiling statistics
    pub async fn get_profiling_statistics(&self) -> Result<ProfilingStatistics> {
        let counters = self.performance_counters.get_current_stats();
        let active_sessions_count = self.active_sessions.lock().len();
        let data_stream_size = self.data_stream.lock().len();
        let config = self.config.read();
        let sampling_interval_ms = (1000.0 / config.sampling_rate) as u64;
        let sample_interval = Duration::from_millis(sampling_interval_ms);
        let total_duration = sample_interval * counters.data_points_processed as u32;
        let buffer_util = data_stream_size as f32 / config.buffer_size as f32;
        let quality_score = if counters.processing_rate > 0 {
            (1.0 - buffer_util.min(1.0)) * 0.5 + 0.5
        } else {
            0.5
        };
        Ok(ProfilingStatistics {
            total_samples: counters.data_points_processed as usize,
            sampling_duration: total_duration,
            average_sample_interval: sample_interval,
            data_quality_score: quality_score as f64,
            active_sessions: active_sessions_count,
            data_points_processed: counters.data_points_processed as usize,
            anomalies_detected: counters.anomalies_detected as usize,
            insights_generated: counters.insights_generated as usize,
            buffer_utilization: buffer_util,
            processing_rate: counters.processing_rate as f64,
            last_updated: Instant::now(),
        })
    }
    /// Start the main profiling loop
    async fn start_profiling_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let profiling_active = Arc::clone(&self.profiling_active);
        let data_stream = Arc::clone(&self.data_stream);
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let data_processor = Arc::clone(&self.data_processor);
        let performance_counters = Arc::clone(&self.performance_counters);
        let handle = tokio::spawn(async move {
            let sampling_interval_ms = (1000.0 / config.sampling_rate) as u64;
            let mut interval = interval(Duration::from_millis(sampling_interval_ms));
            while profiling_active.load(Ordering::Relaxed) {
                interval.tick().await;
                if let Ok(metrics) = metrics_collector.collect_current_metrics().await {
                    let data_point = ProfileDataPoint::from_metrics(metrics);
                    {
                        let mut stream = data_stream.lock();
                        stream.push_back(data_point.clone());
                        if stream.len() > config.buffer_size {
                            stream.pop_front();
                        }
                    }
                    if let Err(e) = data_processor.process_data_point(&data_point).await {
                        eprintln!("Error processing data point: {}", e);
                    }
                    performance_counters.increment_data_points_processed();
                }
            }
        });
        self.control_handles.lock().push(handle);
        Ok(())
    }
    /// Start monitoring for a specific test
    async fn start_test_monitoring(&self, test_id: &str) -> Result<()> {
        let test_id = test_id.to_string();
        let active_sessions = Arc::clone(&self.active_sessions);
        let anomaly_detector = Arc::clone(&self.anomaly_detector);
        let insights_generator = Arc::clone(&self.insights_generator);
        let profiling_active = Arc::clone(&self.profiling_active);
        let data_stream = Arc::clone(&self.data_stream);
        let handle = tokio::spawn(async move {
            while profiling_active.load(Ordering::Relaxed) {
                let session_exists = { active_sessions.lock().contains_key(&test_id) };
                if !session_exists {
                    break;
                }
                let samples: Vec<RealTimeMetrics> = {
                    let stream = data_stream.lock();
                    stream.iter().map(|point| point.metrics.clone()).collect()
                };
                if let Err(e) = anomaly_detector.check_test_anomalies(&test_id, &samples).await {
                    eprintln!("Error checking anomalies for test {}: {}", test_id, e);
                }
                if let Err(e) = insights_generator.update_test_insights(&test_id, &samples).await {
                    eprintln!("Error updating insights for test {}: {}", test_id, e);
                }
                sleep(Duration::from_millis(500)).await;
            }
        });
        self.control_handles.lock().push(handle);
        Ok(())
    }
}
/// Performance counter statistics snapshot
#[derive(Debug, Clone)]
pub struct PerformanceCounterStats {
    pub data_points_processed: u64,
    pub anomalies_detected: u64,
    pub insights_generated: u64,
    pub processing_rate: u64,
}
/// Real-time streaming analysis of profiling data with immediate insights and anomaly detection
///
/// The StreamingAnalyzer processes continuous streams of profiling data to provide
/// real-time insights, pattern detection, and anomaly identification with minimal latency.
#[derive(Debug)]
pub struct StreamingAnalyzer {
    /// Analyzer configuration
    config: Arc<RwLock<StreamingAnalyzerConfig>>,
    /// Analysis state
    analyzing: Arc<AtomicBool>,
    /// Stream processing buffer
    processing_buffer: Arc<Mutex<VecDeque<ProfileDataPoint>>>,
    /// Analysis results cache
    results_cache: Arc<Mutex<BTreeMap<DateTime<Utc>, StreamingAnalysisResult>>>,
    /// Pattern detection engine
    pattern_detector: Arc<RealTimePatternDetector>,
    /// Statistical analyzer
    stats_analyzer: Arc<StreamingStatisticalAnalyzer>,
    /// Analysis handles
    analysis_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl StreamingAnalyzer {
    /// Create a new streaming analyzer
    pub async fn new(config: StreamingAnalyzerConfig) -> Result<Self> {
        let pattern_detector = Arc::new(RealTimePatternDetector::new());
        let stats_analyzer = Arc::new(StreamingStatisticalAnalyzer::new());
        let mut pipelines: Vec<Box<dyn StreamingPipeline + Send + Sync>> = Vec::new();
        pipelines.push(Box::new(PerformanceAnalysisPipeline::new()));
        pipelines.push(Box::new(ResourceAnalysisPipeline::new()));
        pipelines.push(Box::new(ConcurrencyAnalysisPipeline::new()));
        pipelines.push(Box::new(AnomalyDetectionPipeline::new()));
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            analyzing: Arc::new(AtomicBool::new(false)),
            processing_buffer: Arc::new(Mutex::new(VecDeque::new())),
            results_cache: Arc::new(Mutex::new(BTreeMap::new())),
            pattern_detector,
            stats_analyzer,
            analysis_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start streaming analysis
    pub async fn start_analysis(&self) -> Result<()> {
        if self.analyzing.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.analyzing.store(true, Ordering::Relaxed);
        self.pattern_detector.start_detection().await?;
        self.stats_analyzer.start_analysis().await?;
        self.start_analysis_loop().await?;
        Ok(())
    }
    /// Stop streaming analysis
    pub async fn stop_analysis(&self) -> Result<()> {
        self.analyzing.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.analysis_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.pattern_detector.stop_detection().await?;
        self.stats_analyzer.stop_analysis().await?;
        Ok(())
    }
    /// Process data points for streaming analysis
    pub async fn process_data_points(
        &self,
        data_points: &[ProfileDataPoint],
    ) -> Result<StreamingAnalysisResult> {
        let start_time = Instant::now();
        {
            let mut buffer = self.processing_buffer.lock();
            for point in data_points {
                buffer.push_back(point.clone());
            }
            let config = self.config.read();
            while buffer.len() > config.buffer_size {
                buffer.pop_front();
            }
        }
        let analysis_results = HashMap::new();
        // Both consumers see the samples this call was given, not an internal
        // field that nothing ever wrote to.
        let window: Vec<RealTimeMetrics> =
            data_points.iter().map(|point| point.metrics.clone()).collect();
        let patterns = self.pattern_detector.detect_patterns(&window).await?;
        let stats = self.stats_analyzer.analyze_stream(&window).await?;
        let result = StreamingAnalysisResult {
            timestamp: Utc::now(),
            pipeline_results: analysis_results,
            detected_patterns: patterns,
            statistical_summary: stats,
            data_points_analyzed: data_points.len(),
            analysis_duration: start_time.elapsed(),
        };
        self.results_cache.lock().insert(result.timestamp, result.clone());
        Ok(result)
    }
    /// Get recent analysis results
    pub async fn get_recent_results(
        &self,
        duration: Duration,
    ) -> Result<Vec<StreamingAnalysisResult>> {
        let cutoff = Utc::now() - chrono::Duration::from_std(duration)?;
        let cache = self.results_cache.lock();
        Ok(cache.range(cutoff..).map(|(_, result)| result.clone()).collect())
    }
    fn cloned_config(&self) -> StreamingAnalyzerConfig {
        let guard = self.config.read();
        guard.clone()
    }
    /// Start the analysis loop
    async fn start_analysis_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let analyzing = Arc::clone(&self.analyzing);
        let processing_buffer = Arc::clone(&self.processing_buffer);
        let pattern_detector = Arc::clone(&self.pattern_detector);
        let stats_analyzer = Arc::clone(&self.stats_analyzer);
        let results_cache = Arc::clone(&self.results_cache);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.analysis_interval);
            while analyzing.load(Ordering::Relaxed) {
                interval.tick().await;
                let data_points: Vec<ProfileDataPoint> = {
                    let mut buffer = processing_buffer.lock();
                    buffer.drain(..).collect()
                };
                if data_points.is_empty() {
                    continue;
                }
                // Before 0.2.1 the drained points were dropped after a 10ms
                // sleep, so the buffered window never reached an analyzer.
                let window: Vec<RealTimeMetrics> =
                    data_points.iter().map(|point| point.metrics.clone()).collect();
                let patterns = match pattern_detector.detect_patterns(&window).await {
                    Ok(patterns) => patterns,
                    Err(e) => {
                        eprintln!("Error detecting streaming patterns: {}", e);
                        continue;
                    },
                };
                let statistical_summary = match stats_analyzer.analyze_stream(&window).await {
                    Ok(stats) => stats,
                    Err(e) => {
                        eprintln!("Error summarising streaming statistics: {}", e);
                        continue;
                    },
                };
                let result = StreamingAnalysisResult {
                    timestamp: Utc::now(),
                    pipeline_results: HashMap::new(),
                    detected_patterns: patterns,
                    statistical_summary,
                    data_points_analyzed: data_points.len(),
                    analysis_duration: Duration::ZERO,
                };
                results_cache.lock().insert(result.timestamp, result);
            }
        });
        self.analysis_handles.lock().push(handle);
        Ok(())
    }
}
/// High-frequency metrics collection with configurable sampling rates and minimal overhead
///
/// The RealTimeMetricsCollector provides continuous collection of performance metrics
/// with adaptive sampling rates and efficient data structures to minimize impact on
/// test execution performance.
#[derive(Debug)]
pub struct RealTimeMetricsCollector {
    /// Collector configuration
    config: Arc<RwLock<MetricsCollectorConfig>>,
    /// Collection state
    collecting: Arc<AtomicBool>,
    /// Current metrics buffer
    metrics_buffer: Arc<Mutex<VecDeque<RealTimeMetrics>>>,
    /// Performance counters
    collection_counters: Arc<Mutex<CollectionCounters>>,
    /// Collection handles
    collection_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl RealTimeMetricsCollector {
    /// Create a new real-time metrics collector.
    ///
    /// Before 0.2.1 this built a `Vec<Box<dyn ResourceMonitorTrait>>` from the
    /// `monitor_*` config flags and stored it; nothing ever consulted that
    /// vector, and the trait's default `collect_metrics` returned an all-zero
    /// `ResourceMetrics` anyway. The flags now gate which families of key
    /// `sample_host` actually reads from the host.
    pub async fn new(config: MetricsCollectorConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            collecting: Arc::new(AtomicBool::new(false)),
            metrics_buffer: Arc::new(Mutex::new(VecDeque::new())),
            collection_counters: Arc::new(Mutex::new(CollectionCounters::new())),
            collection_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start metrics collection
    pub async fn start_collection(&self) -> Result<()> {
        if self.collecting.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.collecting.store(true, Ordering::Relaxed);
        self.start_collection_loop().await?;
        Ok(())
    }
    /// Stop metrics collection
    pub async fn stop_collection(&self) -> Result<()> {
        self.collecting.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.collection_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        Ok(())
    }
    /// Collect a metrics snapshot from the running host.
    ///
    /// Before 0.2.1 this returned `RealTimeMetrics::new()` -- an empty map with
    /// a fresh timestamp -- so every downstream consumer (the streaming
    /// analyzer, the anomaly detectors, the insight engines) reasoned over a
    /// sample window that carried no measurements at all. Each key is now read
    /// from `sysinfo` at call time; a reading the platform does not report is
    /// omitted from the map rather than substituted with zero.
    pub async fn collect_current_metrics(&self) -> Result<RealTimeMetrics> {
        let metrics = Self::sample_host(&self.cloned_config()).await;
        {
            let mut buffer = self.metrics_buffer.lock();
            buffer.push_back(metrics.clone());
            if buffer.len() > DEFAULT_METRICS_BUFFER_SIZE {
                buffer.pop_front();
            }
        }
        self.collection_counters.lock().increment_collections();
        Ok(metrics)
    }
    /// Get recent metrics history
    pub async fn get_recent_metrics(&self, count: usize) -> Result<Vec<RealTimeMetrics>> {
        let buffer = self.metrics_buffer.lock();
        let start_index = buffer.len().saturating_sub(count);
        Ok(buffer.range(start_index..).cloned().collect())
    }
    fn cloned_config(&self) -> MetricsCollectorConfig {
        let guard = self.config.read();
        guard.clone()
    }
    /// Start the collection loop.
    ///
    /// Before 0.2.1 this loop cloned four `Arc`s into `_`-prefixed bindings and
    /// then slept for 100ms per tick without touching any of them, so
    /// `collecting == true` never produced a single sample.
    async fn start_collection_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let collecting = Arc::clone(&self.collecting);
        let metrics_buffer = Arc::clone(&self.metrics_buffer);
        let collection_counters = Arc::clone(&self.collection_counters);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.collection_interval);
            while collecting.load(Ordering::Relaxed) {
                interval.tick().await;
                let sample = Self::sample_host(&config).await;
                {
                    let mut buffer = metrics_buffer.lock();
                    buffer.push_back(sample);
                    if buffer.len() > DEFAULT_METRICS_BUFFER_SIZE {
                        buffer.pop_front();
                    }
                }
                collection_counters.lock().increment_collections();
            }
        });
        self.collection_handles.lock().push(handle);
        Ok(())
    }

    /// Read one host snapshot; shared by the loop and `collect_current_metrics`.
    ///
    /// CPU utilization needs two `sysinfo` samples separated by
    /// `MINIMUM_CPU_UPDATE_INTERVAL`; `measure_host_async` performs both off the
    /// runtime, so a single-refresh zero is never reported as an idle CPU.
    /// Each `monitor_*` flag gates the family of keys it names, and a reading
    /// the platform does not report is omitted rather than substituted.
    async fn sample_host(config: &MetricsCollectorConfig) -> RealTimeMetrics {
        let mut metrics = RealTimeMetrics::new();
        metrics.timestamp = Utc::now();
        let host = crate::server::system_stats::measure_host_async().await;

        if config.monitor_cpu {
            metrics.metrics.insert("cpu_utilization_percent".to_string(), host.cpu_percent);
            let load = sysinfo::System::load_average();
            if load.one > 0.0 {
                metrics.metrics.insert("load_average_one".to_string(), load.one);
            }
            if let Ok(parallelism) = std::thread::available_parallelism() {
                metrics.metrics.insert(
                    "available_parallelism".to_string(),
                    parallelism.get() as f64,
                );
            }
        }
        if config.monitor_memory && host.total_memory_bytes > 0 {
            metrics.metrics.insert(
                "memory_total_bytes".to_string(),
                host.total_memory_bytes as f64,
            );
            metrics.metrics.insert(
                "memory_used_bytes".to_string(),
                host.used_memory_bytes as f64,
            );
            metrics.metrics.insert(
                "memory_utilization".to_string(),
                host.memory_percent / 100.0,
            );
            metrics.metrics.insert(
                "process_memory_bytes".to_string(),
                host.process_memory_bytes as f64,
            );
        }
        if config.monitor_io {
            if let Some(disk_percent) = host.disk_percent {
                metrics.metrics.insert("disk_utilization_percent".to_string(), disk_percent);
            }
        }
        // `config.monitor_network` gates nothing: reading interface counters
        // needs `sysinfo`'s `network` feature, which the workspace does not
        // enable (see `manager::performance_coordinator`). Emitting a zeroed
        // network key would be indistinguishable from an idle link.
        metrics
    }
}
/// Dynamic optimization system that adjusts profiling strategies based on real-time feedback
///
/// The AdaptiveOptimizer continuously monitors profiling performance and automatically
/// adjusts strategies, sampling rates, and resource allocation to optimize profiling
/// effectiveness while minimizing overhead.
#[derive(Debug)]
pub struct AdaptiveOptimizer {
    /// Optimizer configuration
    config: Arc<RwLock<AdaptiveOptimizerConfig>>,
    /// Optimization strategies
    strategies: Arc<Mutex<Vec<Box<dyn OptimizationStrategy + Send + Sync>>>>,
    /// Optimization state
    optimizing: Arc<AtomicBool>,
    /// Performance metrics tracker
    performance_tracker: Arc<OptimizationPerformanceTracker>,
    /// Strategy effectiveness analyzer
    effectiveness_analyzer: Arc<StrategyEffectivenessAnalyzer>,
    /// Current optimization context
    optimization_context: Arc<RwLock<OptimizationContext>>,
    /// Optimization history
    optimization_history: Arc<Mutex<VecDeque<OptimizationEvent>>>,
    /// Optimization handles
    optimization_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub async fn new(config: AdaptiveOptimizerConfig) -> Result<Self> {
        let performance_tracker = Arc::new(OptimizationPerformanceTracker::new());
        let effectiveness_analyzer = Arc::new(StrategyEffectivenessAnalyzer::new());
        let mut strategies: Vec<Box<dyn OptimizationStrategy + Send + Sync>> = Vec::new();
        strategies.push(Box::new(SamplingRateOptimizer::new()));
        strategies.push(Box::new(BufferSizeOptimizer::new(1000, 2000)));
        strategies.push(Box::new(AnalysisWindowOptimizer::new(100, true)));
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            strategies: Arc::new(Mutex::new(strategies)),
            optimizing: Arc::new(AtomicBool::new(false)),
            performance_tracker,
            effectiveness_analyzer,
            optimization_context: Arc::new(RwLock::new(OptimizationContext::default())),
            optimization_history: Arc::new(Mutex::new(VecDeque::new())),
            optimization_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start adaptive optimization
    pub async fn start_optimization(&self) -> Result<()> {
        if self.optimizing.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.optimizing.store(true, Ordering::Relaxed);
        self.performance_tracker.start_tracking()?;
        self.effectiveness_analyzer.start_analysis()?;
        self.start_optimization_loop().await?;
        Ok(())
    }
    /// Stop adaptive optimization
    pub async fn stop_optimization(&self) -> Result<()> {
        self.optimizing.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.optimization_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.performance_tracker.stop_tracking()?;
        self.effectiveness_analyzer.stop_analysis()?;
        Ok(())
    }
    /// Apply optimization based on current performance data
    pub async fn apply_optimization(
        &self,
        performance_data: &OptimizationPerformanceData,
    ) -> Result<OptimizationResult> {
        self.update_optimization_context(performance_data).await?;
        let effectiveness = self.effectiveness_analyzer.analyze_current_effectiveness()?;
        let context = self.cloned_optimization_context();
        let mut optimization_results = Vec::new();
        let strategies = self.strategies.lock();
        for strategy in strategies.iter() {
            if strategy.is_applicable(&context) {
                match strategy.apply_optimization(performance_data).await {
                    Ok(result) => optimization_results.push(result),
                    Err(e) => eprintln!("Error applying optimization strategy: {}", e),
                }
            }
        }
        // Record each strategy's observed effectiveness so the analyzer has
        // something real to average on the next pass.
        for result in &optimization_results {
            self.effectiveness_analyzer
                .record_effectiveness(&result.strategy_name, result.effectiveness_score);
        }
        let timestamp = Utc::now();
        let event_id = format!("opt_event_{}", timestamp.timestamp_nanos_opt().unwrap_or(0));
        let optimization_id = format!("opt_{}", timestamp.timestamp_millis());
        let mut event_data = HashMap::new();
        event_data.insert(
            "result_count".to_string(),
            optimization_results.len().to_string(),
        );
        event_data.insert("timestamp".to_string(), timestamp.to_rfc3339());
        for (idx, result) in optimization_results.iter().enumerate() {
            event_data.insert(format!("strategy_{}", idx), result.strategy_name.clone());
        }
        let effectiveness_score = if !effectiveness.is_empty() {
            effectiveness.values().sum::<f64>() / effectiveness.len() as f64
        } else {
            0.0
        };
        let optimization_event = OptimizationEvent {
            event_id,
            event_type: "adaptive_optimization_applied".to_string(),
            timestamp,
            optimization_id,
            event_data,
            applied_strategies: optimization_results
                .iter()
                .map(|r| r.strategy_name.clone())
                .collect(),
            effectiveness_score,
            performance_improvement: self.calculate_performance_improvement(&optimization_results),
        };
        self.optimization_history.lock().push_back(optimization_event);
        let results: Vec<OptimizationResult> =
            optimization_results.into_iter().map(|r| r.result).collect();
        Ok(OptimizationResult::combined(results))
    }
    /// Get optimization recommendations
    pub async fn get_optimization_recommendations(
        &self,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let effectiveness = self.effectiveness_analyzer.analyze_current_effectiveness()?;
        let context = self.cloned_optimization_context();
        let mut recommendations = Vec::new();
        let strategies = self.strategies.lock();
        for strategy in strategies.iter() {
            if let Ok(recommendation) = strategy.get_recommendation(&context, &effectiveness).await
            {
                recommendations.push(recommendation);
            }
        }
        recommendations.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.expected_benefit
                        .partial_cmp(&a.expected_benefit)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        Ok(recommendations)
    }
    /// Update optimization context with new performance data
    async fn update_optimization_context(
        &self,
        performance_data: &OptimizationPerformanceData,
    ) -> Result<()> {
        let mut context = self.optimization_context.write();
        let previous_performance = context.current_performance;
        context.current_performance = performance_data.overall_score;
        context.last_updated = Utc::now();
        context.optimization_cycles += 1;
        if performance_data.overall_score > previous_performance {
            context.performance_trend = "improving".to_string();
        } else if performance_data.overall_score < previous_performance {
            context.performance_trend = "degrading".to_string();
        } else {
            context.performance_trend = "stable".to_string();
        }
        Ok(())
    }
    /// Calculate performance improvement from optimization results
    fn calculate_performance_improvement(&self, results: &[StrategyOptimizationResult]) -> f64 {
        results.iter().map(|r| r.effectiveness_score).sum::<f64>() / results.len() as f64
    }
    fn cloned_config(&self) -> AdaptiveOptimizerConfig {
        let guard = self.config.read();
        guard.clone()
    }
    fn cloned_optimization_context(&self) -> OptimizationContext {
        let guard = self.optimization_context.read();
        guard.clone()
    }
    /// Start the optimization loop
    async fn start_optimization_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let optimizing = Arc::clone(&self.optimizing);
        let performance_tracker = Arc::clone(&self.performance_tracker);
        let optimization_context = Arc::clone(&self.optimization_context);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.optimization_interval);
            while optimizing.load(Ordering::Relaxed) {
                interval.tick().await;
                // Before 0.2.1 this built an `OptimizationPerformanceData` with
                // `overall_score: 0.0` and immediately dropped it with
                // `let _ = performance_data;`. The loop now publishes the
                // measured score into the shared optimization context, which
                // `apply_optimization` and the recommendations path both read.
                let Ok(metrics) = performance_tracker.get_current_performance() else {
                    continue;
                };
                let score = Self::score_metrics(&metrics);
                let mut context = optimization_context.write();
                context.current_performance = score;
                context.last_updated = Utc::now();
            }
        });
        self.optimization_handles.lock().push(handle);
        Ok(())
    }
    /// Collapse a performance sample into a single 0..1 score.
    ///
    /// Only the two rates the sample genuinely carries contribute: the error
    /// rate (lower is better) and the mean response time relative to a one
    /// second reference. A sample carrying neither scores zero, which is
    /// distinguishable from a scored sample by `recorded_sample_count`.
    fn score_metrics(metrics: &PerformanceMetrics) -> f64 {
        let error_component = (1.0 - metrics.error_rate).clamp(0.0, 1.0);
        let latency_component = if metrics.average_response_time_ms > 0.0 {
            (1.0 - metrics.average_response_time_ms / 1000.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (error_component + latency_component) / 2.0
    }
}
/// Active profiling session information
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    pub test_id: String,
    pub start_time: DateTime<Utc>,
    pub status: ProfilingStatus,
    pub metrics_collected: u64,
    pub anomalies_detected: u64,
    pub insights_generated: u64,
}
impl ProfilingSession {
    pub async fn new(test_id: String) -> Result<Self> {
        Ok(Self {
            test_id,
            start_time: Utc::now(),
            status: ProfilingStatus::active(),
            metrics_collected: 0,
            anomalies_detected: 0,
            insights_generated: 0,
        })
    }
}
/// Real-time trend analysis and prediction for performance metrics
///
/// The PerformanceTrendAnalyzer continuously monitors performance trends,
/// predicts future behavior, and identifies emerging patterns in real-time
/// profiling data with statistical and machine learning approaches.
#[derive(Debug)]
pub struct PerformanceTrendAnalyzer {
    /// Analyzer configuration
    config: Arc<RwLock<TrendAnalyzerConfig>>,
    /// Analysis state
    analyzing: Arc<AtomicBool>,
    /// Trend analysis algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn TrendAnalysisAlgorithm + Send + Sync>>>>,
    /// Time series database for historical data
    time_series_db: Arc<TimeSeriesDatabase>,
    /// Prediction models
    prediction_models: Arc<RwLock<HashMap<String, PredictionModel>>>,
    /// Trend detection engine
    trend_detector: Arc<TrendDetectionEngine>,
    /// Current trend analysis results
    current_trends: Arc<RwLock<HashMap<String, TrendAnalysisResult>>>,
    /// Analysis handles
    analysis_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl PerformanceTrendAnalyzer {
    /// Create a new performance trend analyzer
    pub async fn new(config: TrendAnalyzerConfig) -> Result<Self> {
        let time_series_db = Arc::new(TimeSeriesDatabase::new(config.database_config.clone()));
        let trend_detector = Arc::new(TrendDetectionEngine::new());
        let mut algorithms: Vec<Box<dyn TrendAnalysisAlgorithm + Send + Sync>> = Vec::new();
        algorithms.push(Box::new(LinearTrendAnalyzer::new()));
        algorithms.push(Box::new(ExponentialTrendAnalyzer::new()));
        algorithms.push(Box::new(SeasonalTrendAnalyzer::new()));
        algorithms.push(Box::new(ArimaTrendAnalyzer::new()));
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            analyzing: Arc::new(AtomicBool::new(false)),
            algorithms: Arc::new(Mutex::new(algorithms)),
            time_series_db,
            prediction_models: Arc::new(RwLock::new(HashMap::new())),
            trend_detector,
            current_trends: Arc::new(RwLock::new(HashMap::new())),
            analysis_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start trend analysis
    pub async fn start_analysis(&self) -> Result<()> {
        if self.analyzing.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.analyzing.store(true, Ordering::Relaxed);
        self.time_series_db.start_collection().await?;
        self.trend_detector.start_detection()?;
        self.start_analysis_loop().await?;
        Ok(())
    }
    /// Stop trend analysis
    pub async fn stop_analysis(&self) -> Result<()> {
        self.analyzing.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.analysis_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.time_series_db.stop_collection().await?;
        self.trend_detector.stop_detection()?;
        Ok(())
    }
    /// Analyze current performance trends
    pub async fn analyze_current_trends(&self) -> Result<HashMap<String, TrendAnalysisResult>> {
        let mut trend_results = HashMap::new();
        let algorithms = self.algorithms.lock();
        let time_series_data = self.time_series_db.get_recent_data(3600).await?;
        let now = Instant::now();
        let current_time = Utc::now();
        let time_series_instant: Vec<(Instant, f64)> = time_series_data
            .iter()
            .map(|(dt, value)| {
                let duration_from_now = current_time.signed_duration_since(*dt);
                let instant = now
                    - std::time::Duration::from_secs(duration_from_now.num_seconds().max(0) as u64);
                (instant, *value)
            })
            .collect();
        for algorithm in algorithms.iter() {
            let trend_analysis = algorithm.analyze_trend(&time_series_instant)?;
            let trend_result = TrendAnalysisResult {
                result_id: format!("trend_{}_{}", algorithm.name(), Utc::now().timestamp()),
                trends: trend_analysis.detected_trends,
                analysis_timestamp: Utc::now(),
                confidence_score: trend_analysis.confidence,
            };
            trend_results.insert(algorithm.name().to_string(), trend_result);
        }
        *self.current_trends.write() = trend_results.clone();
        Ok(trend_results)
    }
    /// Predict future performance based on current trends
    pub async fn predict_future_performance(
        &self,
        horizon: Duration,
    ) -> Result<PerformancePrediction> {
        let trends = self.current_trends_snapshot();
        let models = self.prediction_models.read();
        let mut predictions = Vec::new();
        let mut predicted_metrics = HashMap::new();
        for (metric_name, model) in models.iter() {
            if let Some(trend) = trends.get(metric_name) {
                let data_points: Vec<f64> = trend
                    .trends
                    .iter()
                    .flat_map(|t| t.data_points.iter().map(|(_, value)| *value))
                    .collect();
                let prediction = model.predict(&data_points)?;
                let predicted_value = prediction.first().copied().unwrap_or(0.0);
                predicted_metrics.insert(metric_name.clone(), predicted_value);
                let metric_prediction = MetricPrediction {
                    metric_name: metric_name.clone(),
                    predicted_value,
                    prediction_confidence: trend.confidence_score,
                    prediction_horizon: horizon,
                    prediction_method: "statistical_ml".to_string(),
                };
                predictions.push(metric_prediction);
            }
        }
        let confidence = self.calculate_prediction_confidence(&predictions);
        Ok(PerformancePrediction {
            predicted_metrics: predicted_metrics.clone(),
            prediction_confidence: confidence,
            prediction_time_horizon: horizon,
            prediction_model: "ensemble_statistical_ml".to_string(),
            horizon,
            predictions: predicted_metrics,
            confidence_level: confidence,
            generated_at: Instant::now(),
        })
    }
    /// Calculate prediction confidence
    fn calculate_prediction_confidence(&self, predictions: &[MetricPrediction]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }
        predictions.iter().map(|p| p.prediction_confidence).sum::<f64>() / predictions.len() as f64
    }
    fn cloned_config(&self) -> TrendAnalyzerConfig {
        let guard = self.config.read();
        guard.clone()
    }
    fn current_trends_snapshot(&self) -> HashMap<String, TrendAnalysisResult> {
        let guard = self.current_trends.read();
        guard.clone()
    }
    /// Start the analysis loop
    async fn start_analysis_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let analyzing = Arc::clone(&self.analyzing);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.analysis_interval);
            while analyzing.load(Ordering::Relaxed) {
                interval.tick().await;
            }
        });
        self.analysis_handles.lock().push(handle);
        Ok(())
    }
}
