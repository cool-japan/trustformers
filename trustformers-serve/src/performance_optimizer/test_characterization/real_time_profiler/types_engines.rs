//! Real-time profiler: anomaly detection, reporting, insights, strategy
//! switching and data processing.
//!
//! Split out of `types.rs` in 0.2.1 to keep that file under the 2000-line
//! limit. `types.rs` retains the profiler itself, the streaming analyzer, the
//! metrics collector and the adaptive optimizer.

use super::super::profiling_pipeline::DataAggregationEngine;
use super::super::types::*;
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::task::JoinHandle;
use tokio::time::interval;

/// Real-time anomaly detection with adaptive thresholds and alerting
///
/// The AnomalyDetectionEngine continuously monitors profiling data streams to
/// identify performance anomalies, resource usage spikes, and behavioral deviations
/// with adaptive thresholds and intelligent alerting mechanisms.
#[derive(Debug)]
pub struct AnomalyDetectionEngine {
    /// Detection configuration
    config: Arc<RwLock<AnomalyDetectionConfig>>,
    /// Detection state
    detecting: Arc<AtomicBool>,
    /// Anomaly detectors
    detectors: Arc<Mutex<Vec<Box<dyn AnomalyDetector + Send + Sync>>>>,
    /// Adaptive threshold manager
    threshold_manager: Arc<AdaptiveThresholdManager>,
    /// Anomaly alert system
    alert_system: Arc<AnomalyAlertSystem>,
    /// Detection history
    detection_history: Arc<Mutex<VecDeque<AnomalyDetectionResult>>>,
    /// Statistical models for baseline behavior
    baseline_models: Arc<RwLock<HashMap<String, BaselineModel>>>,
    /// Detection handles
    detection_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl AnomalyDetectionEngine {
    /// Create a new anomaly detection engine
    pub async fn new(config: AnomalyDetectionConfig) -> Result<Self> {
        let threshold_manager = Arc::new(AdaptiveThresholdManager::new());
        let alert_system = Arc::new(AnomalyAlertSystem::new());
        let mut detectors: Vec<Box<dyn AnomalyDetector + Send + Sync>> = Vec::new();
        detectors.push(Box::new(StatisticalAnomalyDetector::new(0.0, 1.0, 3.0)));
        detectors.push(Box::new(ThresholdAnomalyDetector::new()));
        detectors.push(Box::new(TrendAnomalyDetector::new(Vec::new(), 2.0)));
        detectors.push(Box::new(PatternAnomalyDetector::new()));
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            detecting: Arc::new(AtomicBool::new(false)),
            detectors: Arc::new(Mutex::new(detectors)),
            threshold_manager,
            alert_system,
            detection_history: Arc::new(Mutex::new(VecDeque::new())),
            baseline_models: Arc::new(RwLock::new(HashMap::new())),
            detection_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start anomaly detection
    pub async fn start_detection(&self) -> Result<()> {
        if self.detecting.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.detecting.store(true, Ordering::Relaxed);
        self.threshold_manager.start_management().await?;
        self.alert_system.start_alerting()?;
        self.start_detection_loop().await?;
        Ok(())
    }
    /// Stop anomaly detection
    pub async fn stop_detection(&self) -> Result<()> {
        self.detecting.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.detection_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.threshold_manager.stop_management().await?;
        self.alert_system.stop_alerting()?;
        Ok(())
    }
    /// Check for anomalies in the supplied window for `test_id`.
    ///
    /// Before 0.2.1 the baseline was fetched into `let _baseline` and dropped,
    /// the detectors were called with no data at all (and so always returned
    /// nothing), and each wrapped result carried a hardcoded
    /// `detection_confidence: 0.9` / `false_positive_rate: 0.05`.
    pub async fn check_test_anomalies(
        &self,
        test_id: &str,
        samples: &[RealTimeMetrics],
    ) -> Result<Vec<AnomalyDetectionResult>> {
        let baseline = self.fit_baseline(test_id, samples)?;
        let observations = InsightObservations::new(samples);
        let mut anomalies = Vec::new();
        {
            let detectors = self.detectors.lock();
            for detector in detectors.iter() {
                let detected = match detector.detect_anomalies(observations, &baseline) {
                    Ok(detected) => detected,
                    Err(e) => {
                        eprintln!("Anomaly detector {} failed: {}", detector.describe(), e);
                        continue;
                    },
                };
                if detected.is_empty() {
                    continue;
                }
                // Confidence and the false-positive rate are aggregated from the
                // per-anomaly figures the detector actually computed.
                let count = detected.len() as f64;
                let false_positive_rate =
                    detected.iter().map(|a| a.false_positive_probability).sum::<f64>() / count;
                anomalies.push(AnomalyDetectionResult {
                    anomalies_detected: detected,
                    detection_confidence: 1.0 - false_positive_rate,
                    detection_timestamp: Utc::now(),
                    false_positive_rate,
                });
            }
        }
        for detection_result in &anomalies {
            self.detection_history.lock().push_back(detection_result.clone());
        }
        for anomaly_result in &anomalies {
            for anomaly_info in &anomaly_result.anomalies_detected {
                if anomaly_info.severity >= AnomalySeverity::Major {
                    self.alert_system.trigger_alert(anomaly_info)?;
                }
            }
        }
        Ok(anomalies)
    }
    /// Fit (and store) the baseline model for `test_id` from `samples`.
    fn fit_baseline(&self, test_id: &str, samples: &[RealTimeMetrics]) -> Result<BaselineModel> {
        let mut models = self.baseline_models.write();
        let model = models.entry(test_id.to_string()).or_insert_with(BaselineModel::new);
        // The stored model is updated in place; the previous implementation
        // cloned it out first, so any update wrote to a temporary.
        model.update_with_recent_data(samples)?;
        Ok(model.clone())
    }
    fn cloned_config(&self) -> AnomalyDetectionConfig {
        let guard = self.config.read();
        guard.clone()
    }
    /// Start the detection loop
    async fn start_detection_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let detecting = Arc::clone(&self.detecting);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.detection_interval);
            while detecting.load(Ordering::Relaxed) {
                interval.tick().await;
            }
        });
        self.detection_handles.lock().push(handle);
        Ok(())
    }
}
/// Live reporting and dashboard updates with configurable output formats
///
/// The RealTimeReportingEngine generates comprehensive real-time reports,
/// updates live dashboards, and provides configurable output formats for
/// monitoring and analysis of ongoing profiling operations.
#[derive(Debug)]
pub struct RealTimeReportingEngine {
    /// Reporting configuration
    config: Arc<RwLock<ReportingEngineConfig>>,
    /// Reporting state
    reporting: Arc<AtomicBool>,
    /// Report generators
    generators: Arc<Mutex<Vec<Box<dyn ReportGenerator + Send + Sync>>>>,
    /// Dashboard updater
    dashboard_updater: Arc<DashboardUpdater>,
    /// Output formatters
    formatters: Arc<Mutex<HashMap<String, Box<dyn OutputFormatter + Send + Sync>>>>,
    /// Report cache for quick access
    report_cache: Arc<RwLock<HashMap<String, RealTimeReport>>>,
    /// Reporting handles
    reporting_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl RealTimeReportingEngine {
    /// Create a new real-time reporting engine
    pub async fn new(config: ReportingEngineConfig) -> Result<Self> {
        let dashboard_updater = Arc::new(DashboardUpdater::new());
        let mut generators: Vec<Box<dyn ReportGenerator + Send + Sync>> = Vec::new();
        generators.push(Box::new(PerformanceReportGenerator::new()));
        generators.push(Box::new(AnomalyReportGenerator::new()));
        generators.push(Box::new(InsightsReportGenerator::new()));
        generators.push(Box::new(TrendReportGenerator::new()));
        let mut formatters: HashMap<String, Box<dyn OutputFormatter + Send + Sync>> =
            HashMap::new();
        formatters.insert("json".to_string(), Box::new(JsonFormatter::new()));
        formatters.insert("html".to_string(), Box::new(HtmlFormatter::new()));
        formatters.insert("csv".to_string(), Box::new(CsvFormatter::new()));
        formatters.insert(
            "prometheus".to_string(),
            Box::new(PrometheusFormatter::new()),
        );
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            reporting: Arc::new(AtomicBool::new(false)),
            generators: Arc::new(Mutex::new(generators)),
            dashboard_updater,
            formatters: Arc::new(Mutex::new(formatters)),
            report_cache: Arc::new(RwLock::new(HashMap::new())),
            reporting_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start reporting
    pub async fn start_reporting(&self) -> Result<()> {
        if self.reporting.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.reporting.store(true, Ordering::Relaxed);
        self.dashboard_updater.start_updates()?;
        self.start_reporting_loop().await?;
        Ok(())
    }
    /// Stop reporting
    pub async fn stop_reporting(&self) -> Result<()> {
        self.reporting.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.reporting_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.dashboard_updater.stop_updates()?;
        Ok(())
    }
    /// Generate comprehensive real-time report
    pub async fn generate_comprehensive_report(&self) -> Result<RealTimeReport> {
        let mut report = RealTimeReport::new();
        let generators = self.generators.lock();
        for generator in generators.iter() {
            match generator.generate_report() {
                Ok(section) => report.add_section("section", &section),
                Err(e) => eprintln!("Error generating report section: {}", e),
            }
        }
        self.report_cache.write().insert("comprehensive".to_string(), report.clone());
        Ok(report)
    }
    /// Format report in specified format
    pub async fn format_report(&self, report: &RealTimeReport, format: &str) -> Result<String> {
        let formatters = self.formatters.lock();
        if let Some(formatter) = formatters.get(format) {
            formatter.format_report(&report.summary).map_err(|e| anyhow::anyhow!("{}", e))
        } else {
            Err(anyhow::anyhow!("Unknown output format: {}", format))
        }
    }
    fn cloned_config(&self) -> ReportingEngineConfig {
        let guard = self.config.read();
        guard.clone()
    }
    /// Start the reporting loop
    async fn start_reporting_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let reporting = Arc::clone(&self.reporting);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.report_generation_interval);
            while reporting.load(Ordering::Relaxed) {
                interval.tick().await;
            }
        });
        self.reporting_handles.lock().push(handle);
        Ok(())
    }
}
/// Generation of real-time insights and recommendations during test execution
///
/// The LiveInsightsGenerator analyzes streaming profiling data to generate
/// actionable insights, performance recommendations, and optimization suggestions
/// in real-time with minimal latency.
#[derive(Debug)]
pub struct LiveInsightsGenerator {
    /// Generator configuration
    config: Arc<RwLock<InsightsGeneratorConfig>>,
    /// Generation state
    generating: Arc<AtomicBool>,
    /// Insight engines
    insight_engines: Arc<Mutex<Vec<Box<dyn InsightEngine + Send + Sync>>>>,
    /// Recommendation system
    recommendation_system: Arc<RecommendationSystem>,
    /// Insight cache for quick access
    insight_cache: Arc<RwLock<HashMap<String, LiveInsights>>>,
    /// Generation handles
    generation_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl LiveInsightsGenerator {
    /// Create a new live insights generator
    pub async fn new(config: InsightsGeneratorConfig) -> Result<Self> {
        let recommendation_system = Arc::new(RecommendationSystem::new());
        let mut insight_engines: Vec<Box<dyn InsightEngine + Send + Sync>> = Vec::new();
        insight_engines.push(Box::new(PerformanceInsightEngine::new()));
        insight_engines.push(Box::new(ResourceInsightEngine::new()));
        insight_engines.push(Box::new(ConcurrencyInsightEngine::new()));
        insight_engines.push(Box::new(OptimizationInsightEngine::new()));
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            generating: Arc::new(AtomicBool::new(false)),
            insight_engines: Arc::new(Mutex::new(insight_engines)),
            recommendation_system,
            insight_cache: Arc::new(RwLock::new(HashMap::new())),
            generation_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start insights generation
    pub async fn start_generation(&self) -> Result<()> {
        if self.generating.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.generating.store(true, Ordering::Relaxed);
        self.recommendation_system.start_recommendations()?;
        self.start_generation_loop().await?;
        Ok(())
    }
    /// Stop insights generation
    pub async fn stop_generation(&self) -> Result<()> {
        self.generating.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.generation_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.recommendation_system.stop_recommendations()?;
        Ok(())
    }
    /// Generate current insights from the supplied observation window.
    ///
    /// Before 0.2.1 this ran every engine and threw the result away
    /// (`Ok(_engine_insights) => {}`), so no engine finding ever reached a
    /// caller; the returned `LiveInsights` held only the recommendation
    /// descriptions.
    pub async fn generate_current_insights(
        &self,
        samples: &[RealTimeMetrics],
    ) -> Result<LiveInsights> {
        let mut insights = LiveInsights::new();
        let observations = InsightObservations::new(samples);
        {
            let engines = self.insight_engines.lock();
            for engine in engines.iter() {
                match engine.generate_insights(observations) {
                    Ok(engine_insights) => insights.insights.extend(engine_insights),
                    Err(e) => eprintln!("Error generating insights from engine: {}", e),
                }
            }
        }
        let recommendations = self.recommendation_system.generate_recommendations()?;
        for recommendation in recommendations.iter() {
            insights
                .insights
                .push(format!("Recommendation: {}", recommendation.description));
        }
        self.insight_cache.write().insert("current".to_string(), insights.clone());
        Ok(insights)
    }
    /// Update insights for a specific test from the supplied window.
    pub async fn update_test_insights(
        &self,
        test_id: &str,
        samples: &[RealTimeMetrics],
    ) -> Result<()> {
        let insights = self.generate_test_specific_insights(test_id, samples)?;
        self.insight_cache.write().insert(test_id.to_string(), insights);
        Ok(())
    }
    /// Read back the insights cached under `key`, if any.
    pub fn cached_insights(&self, key: &str) -> Option<LiveInsights> {
        self.insight_cache.read().get(key).cloned()
    }
    /// Generate test-specific insights.
    fn generate_test_specific_insights(
        &self,
        test_id: &str,
        samples: &[RealTimeMetrics],
    ) -> Result<LiveInsights> {
        let mut insights = LiveInsights::new();
        let observations = InsightObservations::new(samples);
        let engines = self.insight_engines.lock();
        for engine in engines.iter() {
            match engine.generate_test_insights(test_id, observations) {
                Ok(test_insights) => insights.insights.extend(test_insights),
                Err(e) => eprintln!("Error generating test insights from engine: {}", e),
            }
        }
        Ok(insights)
    }
    fn cloned_config(&self) -> InsightsGeneratorConfig {
        let guard = self.config.read();
        guard.clone()
    }
    /// Start the generation loop
    async fn start_generation_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let generating = Arc::clone(&self.generating);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.generation_interval);
            while generating.load(Ordering::Relaxed) {
                interval.tick().await;
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        });
        self.generation_handles.lock().push(handle);
        Ok(())
    }
}
/// Profile data point containing comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileDataPoint {
    pub timestamp: DateTime<Utc>,
    pub test_id: Option<String>,
    pub point_id: String,
    pub value: f64,
    pub metrics: RealTimeMetrics,
    pub context: ProfilingContext,
}
impl ProfileDataPoint {
    pub fn from_metrics(metrics: RealTimeMetrics) -> Self {
        Self {
            timestamp: Utc::now(),
            test_id: None,
            point_id: format!("point_{}", Utc::now().timestamp_nanos_opt().unwrap_or(0)),
            value: 0.0,
            metrics,
            context: ProfilingContext::default(),
        }
    }
}
/// Intelligent switching between profiling strategies based on observed patterns
///
/// The AdaptiveStrategySwitcher monitors profiling effectiveness and automatically
/// switches between different profiling strategies to optimize resource usage
/// and data quality based on real-time observations.
#[derive(Debug)]
pub struct AdaptiveStrategySwitcher {
    /// Switcher configuration
    config: Arc<RwLock<StrategySwitcherConfig>>,
    /// Switching state
    switching: Arc<AtomicBool>,
    /// Available profiling strategies
    strategies: Arc<Mutex<Vec<Arc<dyn ProfilingStrategy + Send + Sync>>>>,
    /// Current active strategy
    current_strategy: Arc<RwLock<Option<String>>>,
    /// Strategy performance tracker
    performance_tracker: Arc<StrategyPerformanceTracker>,
    /// Strategy selection algorithm
    selection_algorithm: Option<Arc<dyn StrategySelectionAlgorithm + Send + Sync>>,
    /// Switching decision history
    switching_history: Arc<Mutex<VecDeque<StrategySwitch>>>,
    /// Switching handles
    switching_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl AdaptiveStrategySwitcher {
    /// Create a new adaptive strategy switcher
    pub async fn new(config: StrategySwitcherConfig) -> Result<Self> {
        let performance_tracker = Arc::new(StrategyPerformanceTracker::new());
        let selection_algorithm = None;
        let mut strategies: Vec<Arc<dyn ProfilingStrategy + Send + Sync>> = Vec::new();
        strategies.push(Arc::new(HighFrequencyStrategy::new()));
        strategies.push(Arc::new(AdaptiveSamplingStrategy::new()));
        strategies.push(Arc::new(ResourceOptimizedStrategy::new()));
        strategies.push(Arc::new(BalancedStrategy::new()));
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            switching: Arc::new(AtomicBool::new(false)),
            strategies: Arc::new(Mutex::new(strategies)),
            current_strategy: Arc::new(RwLock::new(None)),
            performance_tracker,
            selection_algorithm,
            switching_history: Arc::new(Mutex::new(VecDeque::new())),
            switching_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start strategy switching
    pub async fn start_switching(&self) -> Result<()> {
        if self.switching.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.switching.store(true, Ordering::Relaxed);
        self.performance_tracker.start_tracking()?;
        self.select_initial_strategy().await?;
        self.start_switching_loop().await?;
        Ok(())
    }
    /// Stop strategy switching
    pub async fn stop_switching(&self) -> Result<()> {
        self.switching.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.switching_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.performance_tracker.stop_tracking()?;
        Ok(())
    }
    /// Evaluate and potentially switch strategies
    pub async fn evaluate_strategy_switch(&self) -> Result<Option<StrategySwitch>> {
        let current_performance = self.performance_tracker.get_current_performance()?;
        let current_strategy_name = self.current_strategy_name();
        if let Some(name) = current_strategy_name.as_ref() {
            if let Some(score) = current_performance.get(name) {
                self.performance_tracker.record(name, *score);
            }
        }
        if let Some(current_name) = current_strategy_name {
            let recommended_strategy = if let Some(ref algo) = self.selection_algorithm {
                use crate::performance_optimizer::test_characterization::types::core::SelectionContext;
                let context = SelectionContext {
                    system_state: HashMap::new(),
                    resource_availability: HashMap::new(),
                    objectives: Vec::new(),
                    time_constraints: Duration::from_secs(0),
                    quality_requirements: Default::default(),
                    risk_tolerance: 0.5,
                    historical_context: HashMap::new(),
                    environmental_factors: HashMap::new(),
                    constraint_priorities: HashMap::new(),
                };
                let selection = algo.select_strategy(&context)?;
                selection.selected_strategy
            } else {
                return Ok(None);
            };
            if recommended_strategy != current_name {
                let switch = self.perform_strategy_switch(&recommended_strategy).await?;
                return Ok(Some(switch));
            }
        }
        Ok(None)
    }
    /// Perform strategy switch
    async fn perform_strategy_switch(&self, new_strategy_name: &str) -> Result<StrategySwitch> {
        let old_strategy = self.current_strategy_name();
        // Clone the Arc handles out of the lock before awaiting: the parking_lot
        // MutexGuard is !Send and must not be held across `.await` (this method is
        // reached from inside `tokio::spawn`, which requires a Send future).
        let (to_activate, to_deactivate) = {
            let strategies = self.strategies.lock();
            let to_activate = strategies.iter().find(|s| s.name() == new_strategy_name).cloned();
            let to_deactivate = old_strategy
                .as_deref()
                .and_then(|old_name| strategies.iter().find(|s| s.name() == old_name).cloned());
            (to_activate, to_deactivate)
        };
        if let Some(strategy) = to_activate {
            strategy.activate().await?;
        }
        if let Some(strategy) = to_deactivate {
            strategy.deactivate().await?;
        }
        *self.current_strategy.write() = Some(new_strategy_name.to_string());
        let expected_improvement = self
            .calculate_expected_improvement(old_strategy.as_deref(), new_strategy_name)
            .await;
        let switch_time = Utc::now();
        let switch = StrategySwitch {
            from_strategy: old_strategy.unwrap_or_else(|| "none".to_string()),
            to_strategy: new_strategy_name.to_string(),
            switch_reason: "performance_optimization".to_string(),
            switch_timestamp: switch_time,
            expected_improvement,
            timestamp: Instant::now(),
            reason: "PerformanceOptimization".to_string(),
        };
        self.switching_history.lock().push_back(switch.clone());
        Ok(switch)
    }
    /// Calculate expected improvement from switching strategies
    async fn calculate_expected_improvement(
        &self,
        from_strategy: Option<&str>,
        to_strategy: &str,
    ) -> f64 {
        let history = self.switching_history.lock();
        let from_strategy_str = from_strategy.unwrap_or("");
        let similar_switches: Vec<&StrategySwitch> = history
            .iter()
            .filter(|s| {
                s.from_strategy.as_str() == from_strategy_str && s.to_strategy == to_strategy
            })
            .collect();
        if !similar_switches.is_empty() {
            let avg_improvement: f64 =
                similar_switches.iter().map(|s| s.expected_improvement).sum::<f64>()
                    / similar_switches.len() as f64;
            if avg_improvement > 0.0 {
                return avg_improvement;
            }
        }
        0.10
    }
    /// Select initial strategy
    async fn select_initial_strategy(&self) -> Result<()> {
        // Clone out of the lock before awaiting: the RwLockReadGuard is !Send and
        // must not be held across `perform_strategy_switch(..).await` (this method
        // runs inside `tokio::spawn`, which requires the future to be Send).
        let initial_strategy = self.cloned_config().default_strategy;
        self.perform_strategy_switch(&initial_strategy).await?;
        Ok(())
    }
    fn cloned_config(&self) -> StrategySwitcherConfig {
        let guard = self.config.read();
        guard.clone()
    }
    fn current_strategy_name(&self) -> Option<String> {
        let guard = self.current_strategy.read();
        guard.clone()
    }
    /// Start the switching loop
    async fn start_switching_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let switching = Arc::clone(&self.switching);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.evaluation_interval);
            while switching.load(Ordering::Relaxed) {
                interval.tick().await;
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        });
        self.switching_handles.lock().push(handle);
        Ok(())
    }
}
/// Real-time data processing pipeline with buffering, filtering, and aggregation
///
/// The StreamingDataProcessor manages the flow of profiling data through various
/// processing stages, ensuring efficient buffering, filtering, and aggregation
/// while maintaining data integrity and minimizing latency.
#[derive(Debug)]
pub struct StreamingDataProcessor {
    /// Processor configuration
    config: Arc<RwLock<DataProcessorConfig>>,
    /// Processing state
    processing: Arc<AtomicBool>,
    /// Input buffer for raw data
    input_buffer: Arc<Mutex<VecDeque<ProfileDataPoint>>>,
    /// Processing stages pipeline
    processing_stages: Arc<Mutex<Vec<Box<dyn ProcessingStage + Send + Sync>>>>,
    /// Output buffer for processed data
    output_buffer: Arc<Mutex<VecDeque<ProcessedDataPoint>>>,
    /// Data filtering engine
    filter_engine: Arc<DataFilterEngine>,
    /// Data aggregation engine
    aggregation_engine: Arc<DataAggregationEngine>,
    /// Flow control manager
    flow_control: Arc<FlowControlManager>,
    /// Processing metrics
    processing_metrics: Arc<ProcessingMetrics>,
    /// Processing handles
    processing_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl StreamingDataProcessor {
    /// Create a new streaming data processor
    pub async fn new(config: DataProcessorConfig) -> Result<Self> {
        let filter_engine = Arc::new(DataFilterEngine::new());
        let aggregation_engine = Arc::new(DataAggregationEngine::new(Default::default()).await?);
        let flow_control = Arc::new(FlowControlManager::new());
        let mut processing_stages: Vec<Box<dyn ProcessingStage + Send + Sync>> = Vec::new();
        processing_stages.push(Box::new(DataValidationStage::new()));
        processing_stages.push(Box::new(DataNormalizationStage::new()));
        processing_stages.push(Box::new(DataEnrichmentStage::new()));
        processing_stages.push(Box::new(DataCompressionStage::new()));
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            processing: Arc::new(AtomicBool::new(false)),
            input_buffer: Arc::new(Mutex::new(VecDeque::new())),
            processing_stages: Arc::new(Mutex::new(processing_stages)),
            output_buffer: Arc::new(Mutex::new(VecDeque::new())),
            filter_engine,
            aggregation_engine,
            flow_control,
            processing_metrics: Arc::new(ProcessingMetrics::new()),
            processing_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// Start data processing
    pub async fn start_processing(&self) -> Result<()> {
        if self.processing.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.processing.store(true, Ordering::Relaxed);
        self.filter_engine.start_filtering().await?;
        self.aggregation_engine.start_aggregation().await?;
        self.flow_control.start_control()?;
        self.start_processing_loop().await?;
        Ok(())
    }
    /// Stop data processing
    pub async fn stop_processing(&self) -> Result<()> {
        self.processing.store(false, Ordering::Relaxed);
        // Scoped so the non-`Send` guard cannot survive into the
        // `.await`s that follow (see `RealTimeTestProfiler::stop_profiling`).
        {
            let mut handles = self.processing_handles.lock();
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        self.filter_engine.stop_filtering().await?;
        self.aggregation_engine.stop_aggregation().await?;
        self.flow_control.stop_control()?;
        Ok(())
    }
    /// Process a single data point
    pub async fn process_data_point(
        &self,
        data_point: &ProfileDataPoint,
    ) -> Result<ProcessedDataPoint> {
        self.input_buffer.lock().push_back(data_point.clone());
        self.flow_control.check_flow_control().await?;
        if !self.filter_engine.should_process() {
            return Ok(ProcessedDataPoint::filtered(data_point.clone()));
        }
        let processed_data = data_point.clone();
        let stages = self.processing_stages.lock();
        let mut stage_results = Vec::new();
        for stage in stages.iter() {
            let stage_start = Instant::now();
            let stage_duration = stage_start.elapsed();
            let mut metrics = HashMap::new();
            metrics.insert("input_value".to_string(), data_point.value);
            metrics.insert("output_value".to_string(), data_point.value);
            metrics.insert("value_delta".to_string(), 0.0);
            stage_results.push(StageProcessingResult {
                stage_name: stage.name().to_string(),
                input_data: data_point.clone(),
                output_data: data_point.clone(),
                duration: stage_duration,
                metrics,
            });
        }
        let processing_timestamp = Utc::now();
        let processed_point = ProcessedDataPoint {
            point_id: format!("processed_{}", data_point.point_id),
            timestamp: data_point.timestamp,
            value: processed_data.value,
            processing_method: stages.iter().map(|s| s.name()).collect::<Vec<_>>().join(" -> "),
            quality_score: 1.0,
            original_data: data_point.clone(),
            processed_data,
            processing_timestamp,
            processing_stage_results: stage_results,
        };
        self.output_buffer.lock().push_back(processed_point.clone());
        self.processing_metrics.increment_processed_points();
        Ok(processed_point)
    }
    /// Get processed data points
    pub async fn get_processed_data(&self, count: usize) -> Result<Vec<ProcessedDataPoint>> {
        let mut buffer = self.output_buffer.lock();
        let mut result = Vec::new();
        for _ in 0..count {
            if let Some(point) = buffer.pop_front() {
                result.push(point);
            } else {
                break;
            }
        }
        Ok(result)
    }
    fn cloned_config(&self) -> DataProcessorConfig {
        let guard = self.config.read();
        guard.clone()
    }
    /// Start the processing loop
    async fn start_processing_loop(&self) -> Result<()> {
        let config = self.cloned_config();
        let processing = Arc::clone(&self.processing);
        let input_buffer = Arc::clone(&self.input_buffer);
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.processing_interval);
            while processing.load(Ordering::Relaxed) {
                interval.tick().await;
                let data_points = {
                    let mut buffer = input_buffer.lock();
                    let mut points = Vec::new();
                    for _ in 0..config.batch_size {
                        if let Some(point) = buffer.pop_front() {
                            points.push(point);
                        } else {
                            break;
                        }
                    }
                    points
                };
                for _data_point in data_points {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            }
        });
        self.processing_handles.lock().push(handle);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The counters live in the `types` half of this split module.
    use super::super::types::{PerformanceCounterStats, RealTimePerformanceCounters};

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn next_usize(&mut self, bound: usize) -> usize {
            (self.next_u64() as usize) % bound.max(1)
        }
    }

    // ---- RealTimePerformanceCounters tests ----
    #[test]
    fn test_performance_counters_new() {
        let counters = RealTimePerformanceCounters::new();
        let stats = counters.get_current_stats();
        assert_eq!(stats.data_points_processed, 0);
        assert_eq!(stats.anomalies_detected, 0);
        assert_eq!(stats.insights_generated, 0);
    }

    #[test]
    fn test_performance_counters_increment_data_points() {
        let counters = RealTimePerformanceCounters::new();
        counters.increment_data_points_processed();
        counters.increment_data_points_processed();
        counters.increment_data_points_processed();
        let stats = counters.get_current_stats();
        assert_eq!(stats.data_points_processed, 3);
    }

    #[test]
    fn test_performance_counters_increment_anomalies() {
        let counters = RealTimePerformanceCounters::new();
        counters.increment_anomalies_detected();
        let stats = counters.get_current_stats();
        assert_eq!(stats.anomalies_detected, 1);
    }

    #[test]
    fn test_performance_counters_increment_insights() {
        let counters = RealTimePerformanceCounters::new();
        counters.increment_insights_generated();
        counters.increment_insights_generated();
        let stats = counters.get_current_stats();
        assert_eq!(stats.insights_generated, 2);
    }

    #[test]
    fn test_performance_counters_independent_increments() {
        let counters = RealTimePerformanceCounters::new();
        counters.increment_data_points_processed();
        counters.increment_anomalies_detected();
        counters.increment_insights_generated();
        let stats = counters.get_current_stats();
        assert_eq!(stats.data_points_processed, 1);
        assert_eq!(stats.anomalies_detected, 1);
        assert_eq!(stats.insights_generated, 1);
    }

    #[test]
    fn test_performance_counters_multiple_increments() {
        let counters = RealTimePerformanceCounters::new();
        for _ in 0..100 {
            counters.increment_data_points_processed();
        }
        let stats = counters.get_current_stats();
        assert_eq!(stats.data_points_processed, 100);
    }

    // ---- ProfileDataPoint tests ----
    #[test]
    fn test_profile_data_point_from_metrics() {
        let metrics = RealTimeMetrics::default();
        let point = ProfileDataPoint::from_metrics(metrics);
        assert!(point.test_id.is_none());
        assert!((point.value - 0.0).abs() < f64::EPSILON);
        assert!(point.point_id.starts_with("point_"));
    }

    #[test]
    fn test_profile_data_point_from_metrics_has_timestamp() {
        let before = Utc::now();
        let metrics = RealTimeMetrics::default();
        let point = ProfileDataPoint::from_metrics(metrics);
        let after = Utc::now();
        assert!(point.timestamp >= before);
        assert!(point.timestamp <= after);
    }

    // ---- PerformanceCounterStats construction ----
    #[test]
    fn test_performance_counter_stats_construction() {
        let stats = PerformanceCounterStats {
            data_points_processed: 1000,
            anomalies_detected: 5,
            insights_generated: 20,
            processing_rate: 100,
        };
        assert_eq!(stats.data_points_processed, 1000);
        assert!(stats.anomalies_detected < stats.insights_generated);
    }

    // ---- ProfilingSession construction ----
    #[test]
    fn test_profiling_context_default() {
        let ctx = ProfilingContext::default();
        let formatted = format!("{:?}", ctx);
        assert!(formatted.contains("ProfilingContext"));
    }

    // ---- RealTimeMetrics default ----
    #[test]
    fn test_real_time_metrics_default() {
        let m = RealTimeMetrics::default();
        let formatted = format!("{:?}", m);
        assert!(formatted.contains("RealTimeMetrics"));
    }

    // ---- StrategySwitcherConfig default ----
    #[test]
    fn test_strategy_switcher_config_default() {
        let c = StrategySwitcherConfig::default();
        let formatted = format!("{:?}", c);
        assert!(formatted.contains("StrategySwitcherConfig"));
    }

    // ---- PerformanceCounterStats zero values ----
    #[test]
    fn test_performance_counter_stats_all_zero() {
        let stats = PerformanceCounterStats {
            data_points_processed: 0,
            anomalies_detected: 0,
            insights_generated: 0,
            processing_rate: 0,
        };
        assert_eq!(
            stats.data_points_processed + stats.anomalies_detected + stats.insights_generated,
            0
        );
    }

    // ---- RealTimeProfilerConfig default ----
    #[test]
    fn test_real_time_profiler_config_default() {
        let c = RealTimeProfilerConfig::default();
        let formatted = format!("{:?}", c);
        assert!(formatted.contains("RealTimeProfilerConfig"));
    }

    // ---- LCG-driven counter tests ----
    #[test]
    fn test_lcg_driven_counter_increments() {
        let mut rng = Lcg::new(42);
        let counters = RealTimePerformanceCounters::new();
        let mut expected_data = 0u64;
        let mut expected_anomalies = 0u64;
        let mut expected_insights = 0u64;
        for _ in 0..100 {
            let choice = rng.next_usize(3);
            match choice {
                0 => {
                    counters.increment_data_points_processed();
                    expected_data += 1;
                },
                1 => {
                    counters.increment_anomalies_detected();
                    expected_anomalies += 1;
                },
                _ => {
                    counters.increment_insights_generated();
                    expected_insights += 1;
                },
            }
        }
        let stats = counters.get_current_stats();
        assert_eq!(stats.data_points_processed, expected_data);
        assert_eq!(stats.anomalies_detected, expected_anomalies);
        assert_eq!(stats.insights_generated, expected_insights);
    }

    #[test]
    fn test_lcg_generates_metric_values() {
        let mut rng = Lcg::new(999);
        for _ in 0..50 {
            let v = rng.next_f64() * 100.0;
            assert!((0.0..100.0).contains(&v));
        }
    }

    // ---- Concurrent counter safety ----
    #[test]
    fn test_counters_arc_shared() {
        let counters = Arc::new(RealTimePerformanceCounters::new());
        let c1 = Arc::clone(&counters);
        let c2 = Arc::clone(&counters);
        c1.increment_data_points_processed();
        c2.increment_data_points_processed();
        let stats = counters.get_current_stats();
        assert_eq!(stats.data_points_processed, 2);
    }

    #[test]
    fn test_lcg_determinism() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..50 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_lcg_different_seeds_different_sequence() {
        let mut rng1 = Lcg::new(1);
        let mut rng2 = Lcg::new(2);
        let mut all_same = true;
        for _ in 0..10 {
            if rng1.next_u64() != rng2.next_u64() {
                all_same = false;
                break;
            }
        }
        assert!(!all_same);
    }
}
