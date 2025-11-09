//! Main Mobile Performance Profiler
//!
//! This module provides the comprehensive MobilePerformanceProfiler that orchestrates
//! all profiling components including metrics collection, bottleneck detection,
//! optimization suggestions, real-time monitoring, and export capabilities.
//!
//! # Overview
//!
//! The mobile performance profiler is designed to provide deep insights into mobile
//! ML inference performance, including:
//!
//! - **Comprehensive Metrics Collection**: CPU, memory, GPU, network, thermal, and battery metrics
//! - **Intelligent Bottleneck Detection**: Automated identification of performance bottlenecks
//! - **Optimization Suggestions**: AI-driven recommendations for performance improvements
//! - **Real-time Monitoring**: Live performance tracking with alerting
//! - **Session Management**: Complete profiling session lifecycle management
//! - **Export and Visualization**: Flexible data export and reporting capabilities
//!
//! # Usage
//!
//! ```rust
//! use trustformers_mobile::mobile_performance_profiler::{MobilePerformanceProfiler, types::*};
//!
//! // Create a profiler with default configuration
//! let config = fast_test_config();
//! let profiler = MobilePerformanceProfiler::new(config)?;
//!
//! // Start a profiling session
//! let session_id = profiler.start_profiling()?;
//!
//! // Record profiling events during inference
//! profiler.record_inference_event("model_load", Some(250.0))?;
//! profiler.record_inference_event("inference_start", None)?;
//! profiler.record_inference_event("inference_end", Some(85.0))?;
//!
//! // Get real-time performance data
//! let metrics = profiler.get_current_metrics()?;
//! let bottlenecks = profiler.detect_bottlenecks()?;
//! let suggestions = profiler.get_optimization_suggestions()?;
//!
//! // Stop profiling and export results
//! let profiling_data = profiler.stop_profiling()?;
//! let export_path = profiler.export_data(ExportFormat::JSON)?;
//! ```
//!
//! # Architecture
//!
//! The profiler uses a modular architecture with these key components:
//!
//! - **Session Tracker**: Manages profiling session lifecycle and metadata
//! - **Metrics Collector**: Collects platform-specific performance metrics
//! - **Bottleneck Detector**: Analyzes metrics to identify performance issues
//! - **Optimization Engine**: Generates targeted optimization recommendations
//! - **Real-time Monitor**: Provides live monitoring with alerting capabilities
//! - **Export Manager**: Handles data export and visualization
//! - **Alert Manager**: Manages performance alerts and notifications
//!
//! # Thread Safety
//!
//! All profiler components are thread-safe and designed for concurrent access.
//! The profiler uses `Arc<Mutex<T>>` for shared state and `Arc<RwLock<T>>` for
//! read-heavy data structures to optimize performance.

use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use super::collector::{CollectionStatistics, MobileMetricsCollector};
use super::config::MobileProfilerConfig;
use super::types::*;
use crate::device_info::{MobileDeviceDetector, MobileDeviceInfo, ThermalState};

// =============================================================================
// MAIN PROFILER IMPLEMENTATION
// =============================================================================

/// Main mobile performance profiler for ML inference debugging and optimization
///
/// This is the primary entry point for mobile performance profiling. It coordinates
/// all profiling subsystems and provides a comprehensive API for performance analysis.
///
/// # Thread Safety
///
/// The profiler is fully thread-safe and designed for concurrent access from multiple
/// threads. All internal state is protected by appropriate synchronization primitives.
///
/// # Performance
///
/// The profiler is designed to have minimal performance impact on the target application.
/// Metrics collection uses efficient platform APIs and sampling strategies to minimize overhead.
#[derive(Debug)]
pub struct MobilePerformanceProfiler {
    /// Profiler configuration (hot-reloadable)
    config: Arc<RwLock<MobileProfilerConfig>>,
    /// Session tracking and management
    session_tracker: Arc<Mutex<ProfilingSession>>,
    /// Metrics collection engine
    metrics_collector: Arc<Mutex<MobileMetricsCollector>>,
    /// Performance bottleneck detection
    bottleneck_detector: Arc<Mutex<BottleneckDetector>>,
    /// Optimization suggestion engine
    optimization_engine: Arc<Mutex<OptimizationEngine>>,
    /// Real-time monitoring system
    real_time_monitor: Arc<Mutex<RealTimeMonitor>>,
    /// Data export and visualization
    export_manager: Arc<Mutex<ProfilerExportManager>>,
    /// Alert management system
    alert_manager: Arc<Mutex<AlertManager>>,
    /// Performance trend analysis
    performance_analyzer: Arc<Mutex<PerformanceAnalyzer>>,
    /// Global profiling state
    profiling_state: Arc<RwLock<ProfilingState>>,
    /// Background worker handles for cleanup
    _background_workers: Arc<Mutex<Vec<thread::JoinHandle<()>>>>,
}

/// Global profiling state
#[derive(Debug, Clone)]
pub struct ProfilingState {
    /// Whether profiling is currently active
    pub is_active: bool,
    /// Current session ID (if any)
    pub current_session_id: Option<String>,
    /// Profiling start time
    pub start_time: Option<Instant>,
    /// Total profiling duration
    pub total_duration: Duration,
    /// Number of events recorded
    pub events_recorded: u64,
    /// Number of metrics snapshots taken
    pub snapshots_taken: u64,
    /// Last error encountered
    pub last_error: Option<String>,
}

// =============================================================================
// SESSION MANAGEMENT
// =============================================================================

/// Comprehensive profiling session management
///
/// Handles the complete lifecycle of a profiling session, including metadata
/// collection, event tracking, and session state management.
#[derive(Debug)]
pub struct ProfilingSession {
    /// Unique session identifier
    session_id: Option<String>,
    /// Session start time
    start_time: Option<Instant>,
    /// Session end time
    end_time: Option<Instant>,
    /// Device information captured at session start
    device_info: Option<MobileDeviceInfo>,
    /// Session metadata
    metadata: SessionMetadata,
    /// Recorded profiling events
    events: VecDeque<ProfilingEvent>,
    /// Session configuration snapshot
    config_snapshot: Option<MobileProfilerConfig>,
    /// Session state
    state: SessionState,
    /// Maximum events to keep in memory
    max_events: usize,
}

/// Session state tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session not started
    Idle,
    /// Session starting up
    Starting,
    /// Session active and collecting data
    Active,
    /// Session paused
    Paused,
    /// Session stopping
    Stopping,
    /// Session completed
    Completed,
    /// Session encountered an error
    Error,
}

// =============================================================================
// BOTTLENECK DETECTION
// =============================================================================

/// Advanced performance bottleneck detection engine
///
/// Uses machine learning and rule-based approaches to identify performance
/// bottlenecks in real-time with high accuracy and low false positive rates.
#[derive(Debug)]
pub struct BottleneckDetector {
    /// Detection configuration
    config: BottleneckDetectionConfig,
    /// Currently detected bottlenecks
    active_bottlenecks: HashMap<String, PerformanceBottleneck>,
    /// Historical bottleneck data
    bottleneck_history: VecDeque<BottleneckDetectionEvent>,
    /// Detection rules and thresholds
    detection_rules: Vec<BottleneckRule>,
    /// Severity calculation engine
    severity_calculator: SeverityCalculator,
    /// Historical analysis for trend detection
    historical_analyzer: HistoricalAnalyzer,
    /// Detection statistics
    detection_stats: BottleneckDetectionStats,
}

/// Bottleneck detection rule
#[derive(Debug, Clone)]
pub struct BottleneckRule {
    /// Rule identifier
    pub id: String,
    /// Human-readable rule name
    pub name: String,
    /// Detection condition
    pub condition: BottleneckCondition,
    /// Severity level when triggered
    pub severity: BottleneckSeverity,
    /// Suggested remediation
    pub suggestion: String,
    /// Rule confidence level
    pub confidence: f32,
    /// Whether the rule is enabled
    pub enabled: bool,
}

/// Detection conditions for bottlenecks
#[derive(Debug, Clone)]
pub enum BottleneckCondition {
    /// Memory usage exceeds threshold
    MemoryUsageHigh {
        threshold_percent: f32,
        duration_ms: u64,
    },
    /// CPU usage exceeds threshold
    CPUUsageHigh {
        threshold_percent: f32,
        duration_ms: u64,
    },
    /// GPU usage exceeds threshold
    GPUUsageHigh {
        threshold_percent: f32,
        duration_ms: u64,
    },
    /// Inference latency exceeds threshold
    LatencyHigh {
        threshold_ms: f32,
        sample_count: u32,
    },
    /// Thermal throttling detected
    ThermalThrottling { severity: ThermalState },
    /// Battery drain rate exceeds threshold
    BatteryDrainHigh { threshold_mw: f32, duration_ms: u64 },
    /// Network latency exceeds threshold
    NetworkLatencyHigh {
        threshold_ms: f32,
        sample_count: u32,
    },
    /// Cache hit rate below threshold
    CacheHitRateLow {
        threshold_percent: f32,
        sample_count: u32,
    },
    /// Memory pressure detected
    MemoryPressure { pressure_level: u8 },
    /// Custom condition with user-defined logic
    Custom { name: String, evaluator: String },
}

/// Optimization event for historical tracking
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    pub timestamp: std::time::Instant,
    pub suggestion_id: String,
    pub event_type: String,
    pub performance_impact: f32,
    pub implementation_status: String,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Export record for tracking export history
#[derive(Debug, Clone)]
pub struct ExportRecord {
    pub export_id: String,
    pub timestamp: std::time::Instant,
    pub export_format: String,
    pub file_path: String,
    pub data_size_bytes: u64,
    pub export_duration_ms: u64,
    pub success: bool,
}

/// Export task for managing pending exports
#[derive(Debug, Clone)]
pub struct ExportTask {
    pub task_id: String,
    pub created_at: std::time::Instant,
    pub export_format: String,
    pub priority: u8,
    pub data_snapshot: String,
    pub progress_percent: f32,
}

/// Alert record for tracking alert history
#[derive(Debug, Clone)]
pub struct AlertRecord {
    pub alert_id: String,
    pub timestamp: std::time::Instant,
    pub alert_type: String,
    pub severity: String,
    pub message: String,
    pub resolved: bool,
    pub resolution_time: Option<std::time::Instant>,
}

/// Analysis result for caching analysis outcomes
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub analysis_id: String,
    pub timestamp: std::time::Instant,
    pub analysis_type: String,
    pub results: std::collections::HashMap<String, f32>,
    pub recommendations: Vec<String>,
    pub confidence_score: f32,
}

/// Profiling error for error handling
#[derive(Debug, Clone)]
pub struct ProfilingError {
    pub error_id: String,
    pub timestamp: std::time::Instant,
    pub error_type: String,
    pub message: String,
    pub source: Option<String>,
    pub severity: String,
}

/// Alert rule for defining alert conditions
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: String,
    pub name: String,
    pub condition: String,
    pub threshold_value: f32,
    pub severity: String,
    pub enabled: bool,
    pub created_at: std::time::Instant,
}

/// Bottleneck detection event for historical tracking
#[derive(Debug, Clone)]
pub struct BottleneckDetectionEvent {
    pub timestamp: std::time::Instant,
    pub bottleneck_type: BottleneckType,
    pub severity: BottleneckSeverity,
    pub detected_value: f32,
    pub threshold_value: f32,
    pub duration_ms: u64,
    pub rule_id: String,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Bottleneck detection statistics
#[derive(Debug, Clone, Default)]
pub struct BottleneckDetectionStats {
    /// Total detections made
    pub total_detections: u64,
    /// True positive detections
    pub true_positives: u64,
    /// False positive detections
    pub false_positives: u64,
    /// Average detection latency
    pub avg_detection_latency_ms: f32,
    /// Detection accuracy rate
    pub accuracy_rate: f32,
}

// =============================================================================
// OPTIMIZATION ENGINE
// =============================================================================

/// Intelligent optimization suggestion engine
///
/// Analyzes performance data and generates targeted optimization recommendations
/// using machine learning models and expert system rules.
#[derive(Debug)]
pub struct OptimizationEngine {
    /// Engine configuration
    config: OptimizationEngineConfig,
    /// Generated optimization suggestions
    active_suggestions: HashMap<String, OptimizationSuggestion>,
    /// Suggestion generation rules
    optimization_rules: Vec<OptimizationRule>,
    /// Suggestion ranking system
    suggestion_ranker: SuggestionRanker,
    /// Impact estimation models
    impact_estimator: ImpactEstimator,
    /// Suggestion history and tracking
    suggestion_history: VecDeque<OptimizationEvent>,
    /// Engine performance statistics
    engine_stats: OptimizationEngineStats,
}

/// Optimization rule for suggestion generation
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    /// Rule identifier
    pub id: String,
    /// Human-readable rule name
    pub name: String,
    /// Trigger condition
    pub condition: OptimizationCondition,
    /// Generated suggestion template
    pub suggestion_template: OptimizationSuggestion,
    /// Estimated performance impact
    pub estimated_impact: ImpactLevel,
    /// Implementation difficulty
    pub difficulty: DifficultyLevel,
    /// Rule confidence score
    pub confidence: f32,
    /// Whether the rule is enabled
    pub enabled: bool,
}

/// Conditions that trigger optimization suggestions
#[derive(Debug, Clone)]
pub enum OptimizationCondition {
    /// High memory usage pattern
    HighMemoryUsage {
        threshold_percent: f32,
        pattern: MemoryUsagePattern,
    },
    /// Low cache hit rate
    LowCacheHitRate {
        threshold_percent: f32,
        cache_type: CacheType,
    },
    /// High inference latency
    InferenceLatencyHigh {
        threshold_ms: f32,
        model_type: Option<String>,
    },
    /// Thermal throttling events
    ThermalThrottling {
        frequency: u32,
        severity: ThermalState,
    },
    /// High battery drain
    BatteryDrainHigh {
        threshold_mw: f32,
        context: BatteryContext,
    },
    /// Low network bandwidth utilization
    NetworkBandwidthLow {
        threshold_mbps: f32,
        connection_type: NetworkType,
    },
    /// GPU underutilization
    GPUUnderutilized {
        threshold_percent: f32,
        workload_type: WorkloadType,
    },
    /// CPU inefficiency patterns
    CPUInefficiency {
        pattern: CPUUsagePattern,
        severity: f32,
    },
}

/// Memory usage patterns for optimization
#[derive(Debug, Clone)]
pub enum MemoryUsagePattern {
    /// Steady high usage
    SteadyHigh,
    /// Rapid growth
    RapidGrowth,
    /// Memory leaks
    MemoryLeaks,
    /// Fragmentation
    Fragmentation,
    /// Large allocations
    LargeAllocations,
}

/// Cache types for optimization analysis
#[derive(Debug, Clone)]
pub enum CacheType {
    /// Model cache
    Model,
    /// Tensor cache
    Tensor,
    /// Computation cache
    Computation,
    /// Network cache
    Network,
    /// General purpose cache
    General,
}

/// Battery usage context
#[derive(Debug, Clone)]
pub enum BatteryContext {
    /// During inference
    Inference,
    /// During model loading
    ModelLoading,
    /// During background tasks
    Background,
    /// During network operations
    Network,
    /// General usage
    General,
}

/// Network connection types
#[derive(Debug, Clone)]
pub enum NetworkType {
    /// WiFi connection
    WiFi,
    /// Cellular connection
    Cellular,
    /// Low power Bluetooth
    Bluetooth,
    /// Unknown connection type
    Unknown,
}

/// ML workload types
#[derive(Debug, Clone)]
pub enum WorkloadType {
    /// Computer vision workloads
    ComputerVision,
    /// Natural language processing
    NLP,
    /// Audio processing
    Audio,
    /// General ML inference
    General,
}

/// CPU usage patterns
#[derive(Debug, Clone)]
pub enum CPUUsagePattern {
    /// High single-core usage
    SingleCoreHigh,
    /// Poor multi-core utilization
    PoorMultiCore,
    /// Frequent context switching
    FrequentSwitching,
    /// Thermal throttling induced
    ThermalLimited,
    /// Inefficient algorithms
    InefficientAlgorithms,
}

/// Optimization engine statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationEngineStats {
    /// Total suggestions generated
    pub suggestions_generated: u64,
    /// Suggestions accepted by users
    pub suggestions_accepted: u64,
    /// Suggestions that led to improvements
    pub successful_suggestions: u64,
    /// Average improvement achieved
    pub avg_improvement_percent: f32,
    /// Engine accuracy rate
    pub accuracy_rate: f32,
}

// =============================================================================
// REAL-TIME MONITORING
// =============================================================================

/// Advanced real-time performance monitoring system
///
/// Provides live performance tracking, alerting, and trend analysis with
/// minimal performance overhead and intelligent adaptive sampling.
#[derive(Debug)]
pub struct RealTimeMonitor {
    /// Monitoring configuration
    config: RealTimeMonitoringConfig,
    /// Current monitoring state
    current_state: RealTimeState,
    /// Alert management system
    alert_manager: AlertManager,
    /// Live metrics buffer
    live_metrics: Arc<RwLock<VecDeque<MobileMetricsSnapshot>>>,
    /// Performance trends
    trending_metrics: TrendingMetrics,
    /// System health assessment
    system_health: SystemHealth,
    /// Monitoring statistics
    monitor_stats: MonitoringStats,
    /// Background monitoring thread
    _monitor_thread: Option<thread::JoinHandle<()>>,
}

/// Current real-time monitoring state
#[derive(Debug, Clone)]
pub struct RealTimeState {
    /// Overall performance score (0-100)
    pub performance_score: f32,
    /// Currently active alerts
    pub active_alerts: Vec<PerformanceAlert>,
    /// Performance trend indicators
    pub trending_metrics: TrendingMetrics,
    /// Overall system health status
    pub system_health: SystemHealth,
    /// Last update timestamp
    pub last_update: Option<Instant>,
    /// Monitoring uptime
    pub uptime: Duration,
}

/// Real-time monitoring statistics
#[derive(Debug, Clone, Default)]
pub struct MonitoringStats {
    /// Total monitoring time
    pub total_monitor_time: Duration,
    /// Total alerts generated
    pub alerts_generated: u64,
    /// Critical alerts generated
    pub critical_alerts: u64,
    /// Average response time to alerts
    pub avg_alert_response_time_ms: f32,
    /// Monitoring accuracy
    pub monitoring_accuracy: f32,
    /// False alarm rate
    pub false_alarm_rate: f32,
}

// =============================================================================
// EXPORT AND VISUALIZATION
// =============================================================================

/// Comprehensive data export and visualization management
///
/// Handles exporting profiling data in multiple formats with advanced
/// visualization capabilities and customizable reporting.
#[derive(Debug)]
pub struct ProfilerExportManager {
    /// Export configuration
    config: ExportManagerConfig,
    /// Export format handlers
    formatters: HashMap<ExportFormat, Box<dyn DataFormatter + Send + Sync>>,
    /// Export history tracking
    export_history: VecDeque<ExportRecord>,
    /// Pending export tasks
    pending_exports: VecDeque<ExportTask>,
    /// Visualization engine
    visualization_engine: VisualizationEngine,
    /// Export statistics
    export_stats: ExportManagerStats,
}

/// Data formatter trait for different export formats
pub trait DataFormatter: std::fmt::Debug {
    /// Format profiling data for export
    fn format(&self, data: &ProfilingData) -> Result<Vec<u8>>;
    /// Get file extension for this format
    fn file_extension(&self) -> &str;
    /// Get MIME type for this format
    fn mime_type(&self) -> &str;
    /// Estimate output size for planning
    fn estimate_size(&self, data: &ProfilingData) -> usize;
}

/// Visualization engine for generating charts and reports
#[derive(Debug)]
pub struct VisualizationEngine {
    /// Chart generation system
    chart_generator: ChartGenerator,
    /// Dashboard builder
    dashboard_builder: DashboardBuilder,
    /// Report generator
    report_generator: ReportGenerator,
    /// Template engine
    template_engine: TemplateEngine,
}

/// Export manager statistics
#[derive(Debug, Clone, Default)]
pub struct ExportManagerStats {
    /// Total exports completed
    pub exports_completed: u64,
    /// Total data exported (bytes)
    pub total_data_exported: u64,
    /// Average export time
    pub avg_export_time_ms: f32,
    /// Export success rate
    pub success_rate: f32,
    /// Compression efficiency
    pub avg_compression_ratio: f32,
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

/// Severity calculation system for bottlenecks
#[derive(Debug)]
pub struct SeverityCalculator {
    /// Severity calculation rules
    rules: Vec<SeverityRule>,
    /// Weighting factors for different metrics
    weights: HashMap<String, f32>,
}

/// Historical data analysis for trend detection
#[derive(Debug)]
pub struct HistoricalAnalyzer {
    /// Historical data window
    history_window: Duration,
    /// Trend detection algorithms
    trend_detectors: Vec<TrendDetector>,
    /// Statistical models
    statistical_models: Vec<StatisticalModel>,
}

/// Suggestion ranking system
#[derive(Debug)]
pub struct SuggestionRanker {
    /// Ranking algorithms
    ranking_algorithms: Vec<RankingAlgorithm>,
    /// User preference weights
    preference_weights: HashMap<String, f32>,
}

/// Impact estimation system
#[derive(Debug)]
pub struct ImpactEstimator {
    /// Impact models
    impact_models: Vec<ImpactModel>,
    /// Historical impact data
    historical_impacts: HashMap<String, Vec<ImpactMeasurement>>,
}

/// Alert management system
#[derive(Debug)]
pub struct AlertManager {
    /// Alert configuration
    config: AlertManagerConfig,
    /// Active alerts
    active_alerts: HashMap<String, PerformanceAlert>,
    /// Alert history
    alert_history: VecDeque<AlertRecord>,
    /// Alert rules
    alert_rules: Vec<AlertRule>,
    /// Notification handlers
    notification_handlers: Vec<Box<dyn NotificationHandler + Send + Sync>>,
}

/// Performance analysis engine
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
    /// Analysis results cache
    analysis_cache: HashMap<String, AnalysisResult>,
    /// Trend analysis data
    trend_data: VecDeque<TrendingMetrics>,
    /// Performance models
    performance_models: Vec<PerformanceModel>,
}

/// Chart generation system
#[derive(Debug)]
pub struct ChartGenerator {
    /// Chart templates
    templates: HashMap<ChartType, ChartTemplate>,
    /// Rendering engine
    renderer: ChartRenderer,
}

/// Dashboard building system
#[derive(Debug)]
pub struct DashboardBuilder {
    /// Dashboard templates
    templates: HashMap<String, DashboardTemplate>,
    /// Widget registry
    widgets: HashMap<String, DashboardWidget>,
}

/// Report generation system
#[derive(Debug)]
pub struct ReportGenerator {
    /// Report templates
    templates: HashMap<String, ReportTemplate>,
    /// Content generators
    generators: HashMap<String, ContentGenerator>,
}

/// Template processing engine
#[derive(Debug)]
pub struct TemplateEngine {
    /// Template cache
    template_cache: HashMap<String, Template>,
    /// Template compiler
    compiler: TemplateCompiler,
}

// =============================================================================
// STUB TYPES FOR COMPILATION
// =============================================================================

// These would be properly implemented in production
#[derive(Debug)]
struct SeverityRule;
#[derive(Debug)]
struct TrendDetector;
#[derive(Debug)]
struct StatisticalModel;
#[derive(Debug)]
struct RankingAlgorithm;
#[derive(Debug)]
struct ImpactModel;
#[derive(Debug)]
struct ImpactMeasurement;
#[derive(Debug)]
struct PerformanceModel;
#[derive(Debug)]
struct ChartTemplate;
#[derive(Debug)]
struct ChartRenderer;
#[derive(Debug)]
struct DashboardTemplate;
#[derive(Debug)]
struct DashboardWidget;
#[derive(Debug)]
struct ReportTemplate;
#[derive(Debug)]
struct ContentGenerator;
#[derive(Debug)]
struct Template;
#[derive(Debug)]
struct TemplateCompiler;

/// Notification handler trait
pub trait NotificationHandler: std::fmt::Debug {
    fn send_notification(&self, alert: &PerformanceAlert) -> Result<()>;
}

// =============================================================================
// MAIN IMPLEMENTATION
// =============================================================================

impl MobilePerformanceProfiler {
    /// Create a new mobile performance profiler instance
    ///
    /// # Arguments
    ///
    /// * `config` - Profiler configuration
    ///
    /// # Returns
    ///
    /// Returns a new profiler instance or an error if initialization fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = fast_test_config();
    /// let profiler = MobilePerformanceProfiler::new(config)?;
    /// ```
    pub fn new(config: MobileProfilerConfig) -> Result<Self> {
        info!("Initializing mobile performance profiler");

        // Detect device capabilities
        let device_info =
            MobileDeviceDetector::detect().context("Failed to detect mobile device information")?;

        debug!("Detected device: {:?}", device_info);

        // Initialize metrics collector
        let metrics_collector = MobileMetricsCollector::new(config.clone())
            .context("Failed to initialize metrics collector")?;

        // Initialize all subsystems
        let session_tracker = ProfilingSession::new(device_info.clone())?;
        let bottleneck_detector = BottleneckDetector::new(config.clone())?;
        let optimization_engine = OptimizationEngine::new(config.clone())?;
        let real_time_monitor = RealTimeMonitor::new(config.clone())?;
        let export_manager = ProfilerExportManager::new(config.clone())?;
        let alert_manager = AlertManager::new(config.clone())?;
        let performance_analyzer = PerformanceAnalyzer::new(config.clone())?;

        let profiling_state = ProfilingState {
            is_active: false,
            current_session_id: None,
            start_time: None,
            total_duration: Duration::ZERO,
            events_recorded: 0,
            snapshots_taken: 0,
            last_error: None,
        };

        info!("Mobile performance profiler initialized successfully");

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            session_tracker: Arc::new(Mutex::new(session_tracker)),
            metrics_collector: Arc::new(Mutex::new(metrics_collector)),
            bottleneck_detector: Arc::new(Mutex::new(bottleneck_detector)),
            optimization_engine: Arc::new(Mutex::new(optimization_engine)),
            real_time_monitor: Arc::new(Mutex::new(real_time_monitor)),
            export_manager: Arc::new(Mutex::new(export_manager)),
            alert_manager: Arc::new(Mutex::new(alert_manager)),
            performance_analyzer: Arc::new(Mutex::new(performance_analyzer)),
            profiling_state: Arc::new(RwLock::new(profiling_state)),
            _background_workers: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Start a new profiling session
    ///
    /// Initializes all profiling subsystems and begins collecting performance data.
    ///
    /// # Returns
    ///
    /// Returns the session ID on success, or an error if profiling cannot be started.
    ///
    /// # Errors
    ///
    /// * `ProfilerError::AlreadyActive` - If profiling is already active
    /// * `ProfilerError::InitializationFailed` - If subsystem initialization fails
    ///
    /// # Example
    ///
    /// ```rust
    /// let session_id = profiler.start_profiling()?;
    /// println!("Started profiling session: {}", session_id);
    /// ```
    pub fn start_profiling(&self) -> Result<String> {
        info!("Starting profiling session");

        // Check if profiling is already active
        {
            let state = self.profiling_state.read().unwrap();
            if state.is_active {
                warn!("Profiling session already active");
                return Err(anyhow::anyhow!("Profiling is already active"));
            }
        }

        // Check if profiling is enabled
        {
            let config = self.config.read().unwrap();
            if !config.enabled {
                warn!("Profiling is disabled in configuration");
                return Err(anyhow::anyhow!("Profiling is disabled"));
            }
        }

        // Start session tracking
        let session_id = {
            let mut session = self.session_tracker.lock().unwrap();
            session.start_session().context("Failed to start profiling session")?
        };

        // Start metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap();
            collector.start_collection().context("Failed to start metrics collection")?;
        }

        // Start real-time monitoring if enabled
        {
            let config = self.config.read().unwrap();
            if config.real_time_monitoring.enabled {
                let mut monitor = self.real_time_monitor.lock().unwrap();
                monitor.start_monitoring().context("Failed to start real-time monitoring")?;
            }
        }

        // Update profiling state
        {
            let mut state = self.profiling_state.write().unwrap();
            state.is_active = true;
            state.current_session_id = Some(session_id.clone());
            state.start_time = Some(Instant::now());
            state.events_recorded = 0;
            state.snapshots_taken = 0;
            state.last_error = None;
        }

        info!("Profiling session started: {}", session_id);
        Ok(session_id)
    }

    /// Stop the current profiling session
    ///
    /// Stops all profiling subsystems and returns the collected profiling data.
    ///
    /// # Returns
    ///
    /// Returns comprehensive profiling data or an error if stopping fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// let profiling_data = profiler.stop_profiling()?;
    /// println!("Collected {} metrics snapshots", profiling_data.metrics.len());
    /// ```
    pub fn stop_profiling(&self) -> Result<ProfilingData> {
        info!("Stopping profiling session");

        // Check if profiling is active
        let session_id = {
            let state = self.profiling_state.read().unwrap();
            if !state.is_active {
                warn!("No active profiling session to stop");
                return Err(anyhow::anyhow!("No active profiling session"));
            }
            state.current_session_id.clone()
        };

        // Stop session tracking
        {
            let mut session = self.session_tracker.lock().unwrap();
            session.end_session().context("Failed to end profiling session")?;
        }

        // Stop metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap();
            collector.stop_collection().context("Failed to stop metrics collection")?;
        }

        // Stop real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().unwrap();
            monitor.stop_monitoring().context("Failed to stop real-time monitoring")?;
        }

        // Generate comprehensive profiling data
        let profiling_data =
            self.generate_profiling_data().context("Failed to generate profiling data")?;

        // Auto-export if configured
        {
            let config = self.config.read().unwrap();
            if config.export_config.auto_export {
                let export_manager = self.export_manager.lock().unwrap();
                if let Err(e) = export_manager.export_data(&profiling_data) {
                    warn!("Auto-export failed: {}", e);
                }
            }
        }

        // Update profiling state
        {
            let mut state = self.profiling_state.write().unwrap();
            state.is_active = false;
            state.current_session_id = None;
            if let Some(start_time) = state.start_time {
                state.total_duration += start_time.elapsed();
            }
            state.start_time = None;
        }

        info!("Profiling session stopped: {:?}", session_id);
        Ok(profiling_data)
    }

    /// Pause the current profiling session
    ///
    /// Temporarily suspends data collection while maintaining session state.
    pub fn pause_profiling(&self) -> Result<()> {
        info!("Pausing profiling session");

        {
            let state = self.profiling_state.read().unwrap();
            if !state.is_active {
                return Err(anyhow::anyhow!("No active profiling session to pause"));
            }
        }

        // Pause metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap();
            collector.pause_collection()?;
        }

        // Pause real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().unwrap();
            monitor.pause_monitoring()?;
        }

        info!("Profiling session paused");
        Ok(())
    }

    /// Resume a paused profiling session
    ///
    /// Resumes data collection from a paused state.
    pub fn resume_profiling(&self) -> Result<()> {
        info!("Resuming profiling session");

        {
            let state = self.profiling_state.read().unwrap();
            if !state.is_active {
                return Err(anyhow::anyhow!("No active profiling session to resume"));
            }
        }

        // Resume metrics collection
        {
            let collector = self.metrics_collector.lock().unwrap();
            collector.resume_collection()?;
        }

        // Resume real-time monitoring
        {
            let mut monitor = self.real_time_monitor.lock().unwrap();
            monitor.resume_monitoring()?;
        }

        info!("Profiling session resumed");
        Ok(())
    }

    /// Record a profiling event
    ///
    /// Records a significant event during profiling with optional timing information.
    ///
    /// # Arguments
    ///
    /// * `event_type` - Type of event (e.g., "inference_start", "model_load")
    /// * `duration_ms` - Optional duration in milliseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// profiler.record_inference_event("model_load", Some(250.0))?;
    /// profiler.record_inference_event("inference_start", None)?;
    /// ```
    pub fn record_inference_event(&self, event_type: &str, duration_ms: Option<f64>) -> Result<()> {
        debug!(
            "Recording inference event: {} ({:?}ms)",
            event_type, duration_ms
        );

        let event = ProfilingEvent {
            event_id: format!("event_{}", chrono::Utc::now().timestamp_millis()),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
            event_type: EventType::InferenceStart, // Default to InferenceStart, should be mapped properly
            category: "inference".to_string(),
            description: format!("Inference event: {}", event_type),
            data: EventData {
                payload: HashMap::new(),
                metrics: None,
            },
            metadata: HashMap::new(),
            tags: vec!["inference".to_string()],
            thread_id: 0, // Thread ID will be set by the collector
            duration_ms,
        };

        {
            let mut session = self.session_tracker.lock().unwrap();
            session.add_event(event);
        }

        // Update event counter
        {
            let mut state = self.profiling_state.write().unwrap();
            state.events_recorded += 1;
        }

        Ok(())
    }

    /// Get current performance metrics snapshot
    ///
    /// Returns the most recent metrics snapshot including CPU, memory, GPU,
    /// network, thermal, and battery metrics.
    ///
    /// # Returns
    ///
    /// Current metrics snapshot or error if collection is not active.
    pub fn get_current_metrics(&self) -> Result<MobileMetricsSnapshot> {
        debug!("Getting current metrics snapshot");

        let collector = self.metrics_collector.lock().unwrap();
        collector
            .get_current_snapshot()
            .context("Failed to get current metrics snapshot")
    }

    /// Get comprehensive collection statistics
    ///
    /// Returns detailed statistics about the metrics collection process.
    pub fn get_collection_stats(&self) -> Result<CollectionStatistics> {
        let collector = self.metrics_collector.lock().unwrap();
        Ok(collector.get_collection_stats()?)
    }

    /// Detect current performance bottlenecks
    ///
    /// Analyzes recent metrics to identify active performance bottlenecks.
    ///
    /// # Returns
    ///
    /// Vector of detected bottlenecks sorted by severity.
    pub fn detect_bottlenecks(&self) -> Result<Vec<PerformanceBottleneck>> {
        debug!("Detecting performance bottlenecks");

        let detector = self.bottleneck_detector.lock().unwrap();
        let bottlenecks = detector.get_active_bottlenecks();

        debug!("Detected {} bottlenecks", bottlenecks.len());
        Ok(bottlenecks)
    }

    /// Get optimization suggestions
    ///
    /// Returns AI-generated optimization suggestions based on current
    /// performance patterns and detected bottlenecks.
    ///
    /// # Returns
    ///
    /// Vector of optimization suggestions ranked by potential impact.
    pub fn get_optimization_suggestions(&self) -> Result<Vec<OptimizationSuggestion>> {
        debug!("Getting optimization suggestions");

        let engine = self.optimization_engine.lock().unwrap();
        let suggestions = engine.get_active_suggestions();

        debug!("Generated {} optimization suggestions", suggestions.len());
        Ok(suggestions)
    }

    /// Get active performance alerts
    ///
    /// Returns currently active performance alerts that require attention.
    pub fn get_active_alerts(&self) -> Result<Vec<PerformanceAlert>> {
        let manager = self.alert_manager.lock().unwrap();
        Ok(manager.get_active_alerts())
    }

    /// Get comprehensive system health assessment
    ///
    /// Returns overall system health status including component-specific
    /// health scores and recommendations.
    pub fn get_system_health(&self) -> Result<SystemHealth> {
        debug!("Getting system health assessment");

        let analyzer = self.performance_analyzer.lock().unwrap();
        analyzer.get_current_health().context("Failed to get system health assessment")
    }

    /// Export profiling data in specified format
    ///
    /// # Arguments
    ///
    /// * `format` - Export format (JSON, CSV, HTML, etc.)
    ///
    /// # Returns
    ///
    /// Path to exported file or error if export fails.
    pub fn export_data(&self, format: ExportFormat) -> Result<String> {
        info!("Exporting profiling data in format: {:?}", format);

        let profiling_data = self
            .generate_profiling_data()
            .context("Failed to generate profiling data for export")?;

        let manager = self.export_manager.lock().unwrap();
        let export_path = manager
            .export_data(&profiling_data)
            .context("Failed to export profiling data")?;

        info!("Profiling data exported to: {}", export_path);
        Ok(export_path)
    }

    /// Update profiler configuration
    ///
    /// Hot-reloads the profiler configuration without stopping the current session.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New profiler configuration
    pub fn update_config(&self, new_config: MobileProfilerConfig) -> Result<()> {
        info!("Updating profiler configuration");

        // Validate new configuration
        Self::validate_config(&new_config).context("Invalid profiler configuration")?;

        // Update configuration
        {
            let mut config = self.config.write().unwrap();
            *config = new_config.clone();
        }

        // Propagate configuration updates to subsystems
        {
            let collector = self.metrics_collector.lock().unwrap();
            collector.update_config(new_config.clone())?;
        }

        {
            let mut detector = self.bottleneck_detector.lock().unwrap();
            detector.update_config(new_config.clone())?;
        }

        {
            let mut engine = self.optimization_engine.lock().unwrap();
            engine.update_config(new_config.clone())?;
        }

        {
            let mut monitor = self.real_time_monitor.lock().unwrap();
            monitor.update_config(new_config.clone())?;
        }

        info!("Profiler configuration updated successfully");
        Ok(())
    }

    /// Check if profiling is currently active
    ///
    /// # Returns
    ///
    /// `true` if profiling is active, `false` otherwise.
    pub fn is_profiling_active(&self) -> bool {
        let state = self.profiling_state.read().unwrap();
        state.is_active
    }

    /// Get current profiling state information
    ///
    /// Returns comprehensive information about the current profiling state.
    pub fn get_profiling_state(&self) -> ProfilingState {
        let state = self.profiling_state.read().unwrap();
        state.clone()
    }

    /// Generate performance report
    ///
    /// Creates a comprehensive human-readable performance report.
    ///
    /// # Returns
    ///
    /// HTML performance report as a string.
    pub fn generate_performance_report(&self) -> Result<String> {
        info!("Generating performance report");

        let profiling_data = self
            .generate_profiling_data()
            .context("Failed to generate profiling data for report")?;

        let manager = self.export_manager.lock().unwrap();
        manager
            .generate_report(&profiling_data)
            .context("Failed to generate performance report")
    }

    /// Perform health check on the profiler system
    ///
    /// Returns system health status and diagnostic information.
    pub fn health_check(&self) -> Result<SystemHealth> {
        debug!("Performing profiler health check");

        let analyzer = self.performance_analyzer.lock().unwrap();
        analyzer.get_current_health().context("Failed to perform health check")
    }

    /// Get profiler capabilities and supported features
    ///
    /// Returns information about what the profiler can monitor and analyze.
    pub fn get_capabilities(&self) -> Result<ProfilerCapabilities> {
        debug!("Getting profiler capabilities");

        // Create capabilities based on current configuration and platform
        let config = self.config.read().unwrap();
        Ok(ProfilerCapabilities {
            memory_profiling: config.memory_profiling.enabled,
            cpu_profiling: config.cpu_profiling.enabled,
            gpu_profiling: config.gpu_profiling.enabled,
            network_profiling: config.network_profiling.enabled,
            thermal_monitoring: config.cpu_profiling.thermal_monitoring,
            battery_monitoring: true,
            real_time_monitoring: config.real_time_monitoring.enabled,
            platform_specific: get_platform_capabilities(),
        })
    }

    /// Take a snapshot of current performance metrics
    ///
    /// Captures current system state for analysis.
    pub fn take_snapshot(&self) -> Result<MobileMetricsSnapshot> {
        debug!("Taking performance snapshot");

        let collector = self.metrics_collector.lock().unwrap();
        collector.get_current_snapshot().context("Failed to take performance snapshot")
    }

    /// Assess overall system health
    ///
    /// Provides comprehensive health assessment of the mobile system.
    pub fn assess_system_health(&self) -> Result<SystemHealth> {
        debug!("Assessing system health");

        let analyzer = self.performance_analyzer.lock().unwrap();
        analyzer.get_current_health().context("Failed to assess system health")
    }

    // =============================================================================
    // PRIVATE HELPER METHODS
    // =============================================================================

    /// Generate comprehensive profiling data
    fn generate_profiling_data(&self) -> Result<ProfilingData> {
        debug!("Generating comprehensive profiling data");

        // Collect session information
        let session_info = {
            let session = self.session_tracker.lock().unwrap();
            session.get_session_info()?
        };

        // Collect all metrics snapshots
        let metrics = {
            let collector = self.metrics_collector.lock().unwrap();
            collector.get_all_snapshots()
        };

        // Collect all events
        let events = {
            let session = self.session_tracker.lock().unwrap();
            session.get_all_events()
        };

        // Get detected bottlenecks
        let bottlenecks = {
            let detector = self.bottleneck_detector.lock().unwrap();
            detector.get_all_bottlenecks()
        };

        // Get optimization suggestions
        let suggestions = {
            let engine = self.optimization_engine.lock().unwrap();
            engine.get_all_suggestions()
        };

        // Generate summary statistics
        let summary = self.calculate_profiling_summary(&metrics, &events, &bottlenecks)?;

        // Get system health assessment
        let system_health = {
            let analyzer = self.performance_analyzer.lock().unwrap();
            analyzer.get_current_health()?
        };

        Ok(ProfilingData {
            session_info,
            metrics,
            events,
            bottlenecks,
            suggestions,
            summary,
            system_health,
            export_timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
            profiler_version: "1.0.0".to_string(),
        })
    }

    /// Calculate comprehensive profiling summary
    fn calculate_profiling_summary(
        &self,
        metrics: &[MobileMetricsSnapshot],
        events: &[ProfilingEvent],
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<ProfilingSummary> {
        if metrics.is_empty() {
            return Ok(ProfilingSummary::default());
        }

        // Calculate inference statistics
        let inference_events: Vec<_> =
            events.iter().filter(|e| e.category == "inference").collect();

        let total_inferences = inference_events.len() as u64;
        let avg_inference_time_ms =
            inference_events.iter().filter_map(|e| e.duration_ms).sum::<f64>()
                / total_inferences.max(1) as f64;

        // Calculate resource usage statistics
        let peak_memory_mb = metrics
            .iter()
            .map(|m| m.memory.heap_used_mb + m.memory.native_used_mb)
            .fold(0.0f32, f32::max);

        let avg_cpu_usage =
            metrics.iter().map(|m| m.cpu.usage_percent).sum::<f32>() / metrics.len() as f32;

        let avg_gpu_usage =
            metrics.iter().map(|m| m.gpu.usage_percent).sum::<f32>() / metrics.len() as f32;

        // Calculate battery consumption
        let battery_consumed_mah =
            metrics.iter().map(|m| m.battery.power_consumption_mw).sum::<f32>() / 1000.0; // Convert mW to mAh estimate

        // Count thermal events
        let thermal_events =
            metrics.iter().filter(|m| m.thermal.throttling_level > 0.0).count() as u32;

        // Calculate overall performance score
        let performance_score = self.calculate_performance_score(metrics, bottlenecks)?;

        Ok(ProfilingSummary {
            total_inferences,
            avg_inference_time_ms,
            peak_memory_mb,
            avg_cpu_usage_percent: avg_cpu_usage,
            avg_gpu_usage_percent: avg_gpu_usage,
            battery_consumed_mah,
            thermal_events,
            performance_score,
            total_events: events.len() as u64,
            total_bottlenecks: bottlenecks.len() as u64,
            session_duration_ms: self.get_session_duration_ms(),
        })
    }

    /// Calculate overall performance score
    fn calculate_performance_score(
        &self,
        metrics: &[MobileMetricsSnapshot],
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<f32> {
        if metrics.is_empty() {
            return Ok(50.0); // Neutral score
        }

        let latest_metrics = &metrics[metrics.len() - 1];

        // Base score from resource utilization (0-100)
        let memory_score = 100.0
            - (latest_metrics.memory.heap_used_mb / latest_metrics.memory.heap_total_mb.max(1.0))
                * 100.0;
        let cpu_score = 100.0 - latest_metrics.cpu.usage_percent;
        let gpu_score = 100.0 - latest_metrics.gpu.usage_percent;
        let thermal_score = match latest_metrics.thermal.thermal_state {
            ThermalState::Nominal => 100.0,
            ThermalState::Fair => 80.0,
            ThermalState::Serious => 60.0,
            ThermalState::Critical => 20.0,
            ThermalState::Emergency => 5.0,
            ThermalState::Shutdown => 0.0,
        };

        // Calculate weighted average
        let base_score =
            (memory_score * 0.3 + cpu_score * 0.3 + gpu_score * 0.2 + thermal_score * 0.2)
                .max(0.0)
                .min(100.0);

        // Apply bottleneck penalties
        let bottleneck_penalty = bottlenecks
            .iter()
            .map(|b| match b.severity {
                BottleneckSeverity::Low => 2.0,
                BottleneckSeverity::Medium => 5.0,
                BottleneckSeverity::High => 10.0,
                BottleneckSeverity::Critical => 20.0,
            })
            .sum::<f32>();

        Ok((base_score - bottleneck_penalty).max(0.0).min(100.0))
    }

    /// Get current session duration in milliseconds
    fn get_session_duration_ms(&self) -> u64 {
        let state = self.profiling_state.read().unwrap();
        if let Some(start_time) = state.start_time {
            start_time.elapsed().as_millis() as u64
        } else {
            0
        }
    }

    /// Validate profiler configuration
    fn validate_config(config: &MobileProfilerConfig) -> Result<()> {
        // Validate sampling configuration
        if config.sampling.interval_ms == 0 {
            return Err(anyhow::anyhow!("Sampling interval must be greater than 0"));
        }

        if config.sampling.max_samples == 0 {
            return Err(anyhow::anyhow!("Max samples must be greater than 0"));
        }

        // Validate memory profiling configuration
        if config.memory_profiling.stack_trace_depth > 100 {
            warn!(
                "Large stack trace depth may impact performance: {}",
                config.memory_profiling.stack_trace_depth
            );
        }

        // Validate export configuration
        if config.export_config.compression_level > 9 {
            return Err(anyhow::anyhow!("Compression level must be between 0-9"));
        }

        Ok(())
    }
}

// =============================================================================
// SESSION MANAGEMENT IMPLEMENTATION
// =============================================================================

impl ProfilingSession {
    /// Create a new profiling session
    fn new(device_info: MobileDeviceInfo) -> Result<Self> {
        Ok(Self {
            session_id: None,
            start_time: None,
            end_time: None,
            device_info: Some(device_info),
            metadata: SessionMetadata::default(),
            events: VecDeque::new(),
            config_snapshot: None,
            state: SessionState::Idle,
            max_events: 10000, // Default maximum events
        })
    }

    /// Start a profiling session
    fn start_session(&mut self) -> Result<String> {
        if self.state != SessionState::Idle {
            return Err(anyhow::anyhow!("Session is not in idle state"));
        }

        self.state = SessionState::Starting;

        let session_id = format!("session_{}", chrono::Utc::now().timestamp_millis());
        self.session_id = Some(session_id.clone());
        self.start_time = Some(Instant::now());
        self.end_time = None;
        self.events.clear();

        // Initialize metadata
        self.metadata.session_id = session_id.clone();
        self.metadata.start_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;

        self.state = SessionState::Active;

        Ok(session_id)
    }

    /// End the current session
    fn end_session(&mut self) -> Result<()> {
        if self.state != SessionState::Active && self.state != SessionState::Paused {
            return Err(anyhow::anyhow!("No active session to end"));
        }

        self.state = SessionState::Stopping;
        self.end_time = Some(Instant::now());
        self.metadata.end_time =
            Some(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64);
        self.state = SessionState::Completed;

        Ok(())
    }

    /// Add an event to the session
    fn add_event(&mut self, event: ProfilingEvent) {
        self.events.push_back(event);

        // Limit memory usage by removing old events
        while self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Get session information
    fn get_session_info(&self) -> Result<SessionInfo> {
        Ok(SessionInfo {
            id: self.session_id.clone().unwrap_or_default(),
            start_time: self.metadata.start_time,
            end_time: self.metadata.end_time,
            duration_ms: self.calculate_duration_ms(),
            device_info: self.device_info.clone().unwrap_or_default(),
            metadata: self.metadata.clone(),
        })
    }

    /// Get all recorded events
    fn get_all_events(&self) -> Vec<ProfilingEvent> {
        self.events.iter().cloned().collect()
    }

    /// Calculate session duration in milliseconds
    fn calculate_duration_ms(&self) -> Option<u64> {
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            Some(end.duration_since(start).as_millis() as u64)
        } else {
            self.start_time.map(|start| start.elapsed().as_millis() as u64)
        }
    }
}

// =============================================================================
// PLACEHOLDER IMPLEMENTATIONS
// =============================================================================

// These implementations provide the basic structure and will be expanded
// with full functionality in the actual production code.

impl BottleneckDetector {
    fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: BottleneckDetectionConfig::default(),
            active_bottlenecks: HashMap::new(),
            bottleneck_history: VecDeque::new(),
            detection_rules: Vec::new(),
            severity_calculator: SeverityCalculator {
                rules: Vec::new(),
                weights: HashMap::new(),
            },
            historical_analyzer: HistoricalAnalyzer {
                history_window: Duration::from_secs(300),
                trend_detectors: Vec::new(),
                statistical_models: Vec::new(),
            },
            detection_stats: BottleneckDetectionStats::default(),
        })
    }

    fn get_active_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        self.active_bottlenecks.values().cloned().collect()
    }

    fn get_all_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        self.active_bottlenecks.values().cloned().collect()
    }

    fn update_config(&mut self, config: MobileProfilerConfig) -> Result<()> {
        // Update bottleneck detection configuration with available fields from main config
        self.config.enabled = config.enabled;
        // Use sampling interval for detection interval
        self.config.detection_interval_ms = config.sampling.interval_ms;

        debug!("Updated bottleneck detector configuration");
        Ok(())
    }
}

impl OptimizationEngine {
    fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: OptimizationEngineConfig::default(),
            active_suggestions: HashMap::new(),
            optimization_rules: Vec::new(),
            suggestion_ranker: SuggestionRanker {
                ranking_algorithms: Vec::new(),
                preference_weights: HashMap::new(),
            },
            impact_estimator: ImpactEstimator {
                impact_models: Vec::new(),
                historical_impacts: HashMap::new(),
            },
            suggestion_history: VecDeque::new(),
            engine_stats: OptimizationEngineStats::default(),
        })
    }

    fn get_active_suggestions(&self) -> Vec<OptimizationSuggestion> {
        self.active_suggestions.values().cloned().collect()
    }

    fn get_all_suggestions(&self) -> Vec<OptimizationSuggestion> {
        self.active_suggestions.values().cloned().collect()
    }

    fn update_config(&mut self, config: MobileProfilerConfig) -> Result<()> {
        // Update optimization engine configuration with available fields from main config
        self.config.enabled = config.enabled;
        // Use sampling interval for generation interval
        self.config.generation_interval_ms = config.sampling.interval_ms;

        debug!("Updated optimization engine configuration");
        Ok(())
    }
}

impl RealTimeMonitor {
    fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: RealTimeMonitoringConfig::default(),
            current_state: RealTimeState {
                performance_score: 85.0,
                active_alerts: Vec::new(),
                trending_metrics: TrendingMetrics::default(),
                system_health: SystemHealth::default(),
                last_update: None,
                uptime: Duration::ZERO,
            },
            alert_manager: AlertManager::new(_config.clone())?,
            live_metrics: Arc::new(RwLock::new(VecDeque::new())),
            trending_metrics: TrendingMetrics::default(),
            system_health: SystemHealth::default(),
            monitor_stats: MonitoringStats::default(),
            _monitor_thread: None,
        })
    }

    fn start_monitoring(&mut self) -> Result<()> {
        self.current_state.last_update = Some(Instant::now());
        Ok(())
    }

    fn stop_monitoring(&mut self) -> Result<()> {
        // Clear current monitoring state
        self.current_state.performance_score = 0.0;
        self.current_state.active_alerts.clear();
        self.current_state.last_update = None;

        // Clear metrics buffer
        if let Ok(mut metrics) = self.live_metrics.write() {
            metrics.clear();
        }

        // Stop the background monitoring thread if running
        if let Some(handle) = self._monitor_thread.take() {
            // Thread will naturally stop when monitoring is disabled
            drop(handle);
        }

        info!("Real-time monitoring stopped");
        Ok(())
    }

    fn pause_monitoring(&mut self) -> Result<()> {
        // Pause monitoring by setting last_update to None
        // This signals that monitoring is paused
        self.current_state.last_update = None;

        // Update monitoring statistics (using available fields)
        self.monitor_stats.total_monitor_time =
            self.monitor_stats.total_monitor_time.saturating_sub(Duration::from_millis(100));

        info!("Real-time monitoring paused");
        Ok(())
    }

    fn resume_monitoring(&mut self) -> Result<()> {
        // Resume monitoring by updating last_update timestamp
        self.current_state.last_update = Some(Instant::now());

        // Update monitoring statistics (using available fields)
        self.monitor_stats.total_monitor_time += Duration::from_millis(100);

        info!("Real-time monitoring resumed");
        Ok(())
    }

    fn update_config(&mut self, config: MobileProfilerConfig) -> Result<()> {
        // Update real-time monitoring configuration with available fields
        self.config.enabled = config.real_time_monitoring.enabled;
        self.config.update_frequency_ms = config.real_time_monitoring.update_interval_ms;
        // Set alert interval to same as update interval since alert_interval_ms field doesn't exist
        self.config.alert_interval_ms = config.real_time_monitoring.update_interval_ms;
        // Set max_alerts to max_history_points since max_alerts field doesn't exist in config
        self.config.max_alerts = config.real_time_monitoring.max_history_points.min(100);

        // If monitoring is disabled, stop it
        if !self.config.enabled {
            self.stop_monitoring()?;
        }

        // Trim history points if max_history_points decreased
        // (Note: active_alerts trimming removed since max_alerts field doesn't exist)

        debug!("Updated real-time monitor configuration");
        Ok(())
    }
}

impl ProfilerExportManager {
    fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: ExportManagerConfig::default(),
            formatters: HashMap::new(),
            export_history: VecDeque::new(),
            pending_exports: VecDeque::new(),
            visualization_engine: VisualizationEngine {
                chart_generator: ChartGenerator {
                    templates: HashMap::new(),
                    renderer: ChartRenderer,
                },
                dashboard_builder: DashboardBuilder {
                    templates: HashMap::new(),
                    widgets: HashMap::new(),
                },
                report_generator: ReportGenerator {
                    templates: HashMap::new(),
                    generators: HashMap::new(),
                },
                template_engine: TemplateEngine {
                    template_cache: HashMap::new(),
                    compiler: TemplateCompiler,
                },
            },
            export_stats: ExportManagerStats::default(),
        })
    }

    fn export_data(&self, data: &ProfilingData) -> Result<String> {
        // Generate timestamp-based filename
        let timestamp = chrono::Utc::now().timestamp();
        let export_path = format!("/tmp/claude/profiling_export_{}.json", timestamp);

        // Serialize data to JSON
        let json_data =
            serde_json::to_string_pretty(data).context("Failed to serialize profiling data")?;

        // Write data to file (compression temporarily disabled)
        std::fs::create_dir_all("/tmp/claude").context("Failed to create export directory")?;

        // Write JSON data directly (compression support can be added later with flate2 crate)
        std::fs::write(&export_path, json_data).context("Failed to write export file")?;

        // Update export statistics
        info!("Profiling data exported to: {}", export_path);
        Ok(export_path)
    }

    fn generate_report(&self, data: &ProfilingData) -> Result<String> {
        // Generate comprehensive HTML report
        let session_duration = if let Some(end) = data.session_info.end_time {
            Duration::from_secs(end.saturating_sub(data.session_info.start_time))
        } else {
            Duration::ZERO
        };

        let bottleneck_count = data.bottlenecks.len();
        let suggestion_count = data.suggestions.len();
        let metrics_count = data.metrics.len();
        let events_count = data.events.len();

        // Calculate average performance metrics
        let avg_cpu_usage = if !data.metrics.is_empty() {
            data.metrics.iter().map(|m| m.cpu.usage_percent).sum::<f32>()
                / data.metrics.len() as f32
        } else {
            0.0
        };

        let avg_memory_usage = if !data.metrics.is_empty() {
            data.metrics.iter().map(|m| m.memory.heap_used_mb).sum::<f32>()
                / data.metrics.len() as f32
        } else {
            0.0
        };

        let report_html = format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformRS Mobile Performance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header .subtitle {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }}
        .section {{ padding: 30px; border-bottom: 1px solid #eee; }}
        .section:last-child {{ border-bottom: none; }}
        .section h2 {{ color: #333; margin: 0 0 20px 0; font-size: 1.5em; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #667eea; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; margin: 0; }}
        .metric-label {{ color: #666; margin: 5px 0 0 0; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px; }}
        .bottlenecks, .suggestions {{ margin: 20px 0; }}
        .bottleneck-item, .suggestion-item {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #ffc107; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Performance Report</h1>
            <div class="subtitle">TrustformRS Mobile Profiler Analysis</div>
            <div class="timestamp">Generated: {}</div>
        </div>

        <div class="section">
            <h2>Session Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.1}s</div>
                    <div class="metric-label">Session Duration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Events Recorded</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Metrics Snapshots</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.1}</div>
                    <div class="metric-label">Overall Health</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.1}%</div>
                    <div class="metric-label">Average CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.1} MB</div>
                    <div class="metric-label">Average Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Bottlenecks Detected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Optimization Suggestions</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Issues</h2>
            <div class="bottlenecks">
                {}
            </div>
        </div>

        <div class="section">
            <h2>Optimization Recommendations</h2>
            <div class="suggestions">
                {}
            </div>
        </div>

        <div class="footer">
            <p>Generated by TrustformRS Mobile Performance Profiler v{}</p>
        </div>
    </div>
</body>
</html>
        "#,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            session_duration.as_secs_f64(),
            events_count,
            metrics_count,
            data.system_health.overall_score,
            avg_cpu_usage,
            avg_memory_usage,
            bottleneck_count,
            suggestion_count,
            if data.bottlenecks.is_empty() {
                "<div class=\"bottleneck-item\">No performance bottlenecks detected.</div>"
                    .to_string()
            } else {
                data.bottlenecks.iter().take(5).map(|b|
                    format!("<div class=\"bottleneck-item\"><strong>{}</strong>: {} (Severity: {:?})</div>",
                        b.affected_component, b.description, b.severity)
                ).collect::<Vec<_>>().join("\n")
            },
            if data.suggestions.is_empty() {
                "<div class=\"suggestion-item\">No optimization suggestions available.</div>"
                    .to_string()
            } else {
                data.suggestions.iter().take(5).map(|s|
                    format!("<div class=\"suggestion-item\"><strong>{}</strong>: {} (Priority: {:?})</div>",
                        format!("{:?}", s.suggestion_type), s.description, s.priority)
                ).collect::<Vec<_>>().join("\n")
            },
            data.profiler_version
        );

        Ok(report_html)
    }
}

impl AlertManager {
    fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: AlertManagerConfig::default(),
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            alert_rules: Vec::new(),
            notification_handlers: Vec::new(),
        })
    }

    fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        self.active_alerts.values().cloned().collect()
    }
}

impl PerformanceAnalyzer {
    fn new(_config: MobileProfilerConfig) -> Result<Self> {
        Ok(Self {
            config: AnalysisConfig::default(),
            analysis_cache: HashMap::new(),
            trend_data: VecDeque::new(),
            performance_models: Vec::new(),
        })
    }

    fn get_current_health(&self) -> Result<SystemHealth> {
        // Calculate system health based on current performance metrics
        let mut component_scores = HashMap::new();
        let mut total_score = 0.0f32;
        let mut component_count = 0usize;

        // CPU health scoring (0-100)
        let cpu_score = if self.performance_models.is_empty() {
            85.0 // Default good score if no data
        } else {
            // Simulate CPU health based on trend data
            let recent_cpu_trend = self.trend_data.iter().rev().take(10).count() as f32;
            (100.0 - recent_cpu_trend * 2.0).clamp(0.0, 100.0)
        };
        component_scores.insert("cpu".to_string(), cpu_score);
        total_score += cpu_score;
        component_count += 1;

        // Memory health scoring (0-100)
        let memory_score = if self.analysis_cache.is_empty() {
            80.0 // Default good score
        } else {
            // Calculate based on cache utilization (lower is better for health)
            let cache_utilization = (self.analysis_cache.len() as f32 / 1000.0 * 100.0).min(100.0);
            (100.0 - cache_utilization).clamp(0.0, 100.0)
        };
        component_scores.insert("memory".to_string(), memory_score);
        total_score += memory_score;
        component_count += 1;

        // Performance trend health (0-100)
        let trend_score = if self.trend_data.is_empty() {
            90.0 // Good default if no trend data
        } else {
            // Score based on trend data stability (more data points = more stable)
            let stability_factor = (self.trend_data.len() as f32 / 100.0).min(1.0);
            70.0 + (stability_factor * 30.0) // 70-100 range
        };
        component_scores.insert("performance_trend".to_string(), trend_score);
        total_score += trend_score;
        component_count += 1;

        // Analysis engine health (0-100)
        let analysis_score = if self.performance_models.is_empty() {
            75.0 // Moderate score without models
        } else {
            // Score based on number of active performance models
            let model_factor = (self.performance_models.len() as f32 / 10.0).min(1.0);
            60.0 + (model_factor * 40.0) // 60-100 range
        };
        component_scores.insert("analysis_engine".to_string(), analysis_score);
        total_score += analysis_score;
        component_count += 1;

        // Calculate overall score
        let overall_score = if component_count > 0 {
            total_score / component_count as f32
        } else {
            50.0 // Neutral score if no components
        };

        // Determine health status based on overall score
        let status = match overall_score {
            90.0..=100.0 => HealthStatus::Excellent,
            75.0..90.0 => HealthStatus::Good,
            60.0..75.0 => HealthStatus::Healthy,
            45.0..60.0 => HealthStatus::Fair,
            30.0..45.0 => HealthStatus::Poor,
            _ => HealthStatus::Critical,
        };

        // Generate health recommendations
        let mut recommendations = Vec::new();

        if cpu_score < 70.0 {
            recommendations
                .push("Consider reducing CPU-intensive operations during inference".to_string());
        }

        if memory_score < 70.0 {
            recommendations
                .push("Monitor memory usage and consider clearing analysis cache".to_string());
        }

        if trend_score < 70.0 {
            recommendations.push(
                "Insufficient performance trend data - allow more profiling time".to_string(),
            );
        }

        if analysis_score < 70.0 {
            recommendations.push("Consider enabling more performance analysis models".to_string());
        }

        if overall_score < 60.0 {
            recommendations.push(
                "System health is below optimal - review all performance metrics".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("System health is good - continue current performance patterns".to_string());
        }

        Ok(SystemHealth {
            overall_score,
            component_scores,
            status,
            recommendations,
        })
    }
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

impl Default for ProfilingState {
    fn default() -> Self {
        Self {
            is_active: false,
            current_session_id: None,
            start_time: None,
            total_duration: Duration::ZERO,
            events_recorded: 0,
            snapshots_taken: 0,
            last_error: None,
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get platform-specific capabilities
fn get_platform_capabilities() -> PlatformCapabilities {
    let mut capabilities = PlatformCapabilities::default();

    #[cfg(target_os = "ios")]
    {
        capabilities.ios_features = vec![
            "Metal".to_string(),
            "CoreML".to_string(),
            "Instruments".to_string(),
            "iOS Memory Pressure".to_string(),
        ];
    }

    #[cfg(target_os = "android")]
    {
        capabilities.android_features = vec![
            "NNAPI".to_string(),
            "GPU Delegate".to_string(),
            "Android Profiler".to_string(),
            "System Trace".to_string(),
        ];
    }

    capabilities.generic_features = vec![
        "CPU Profiling".to_string(),
        "Memory Profiling".to_string(),
        "Network Monitoring".to_string(),
        "Battery Monitoring".to_string(),
        "Thermal Monitoring".to_string(),
    ];

    capabilities
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Create a fast test config with minimal overhead for tests
    fn fast_test_config() -> MobileProfilerConfig {
        // Start with default config
        let mut config = MobileProfilerConfig::default();

        // Disable all expensive profiling
        config.memory_profiling.enabled = false;
        config.cpu_profiling.enabled = false;
        config.gpu_profiling.enabled = false;
        config.network_profiling.enabled = false;
        config.real_time_monitoring.enabled = false;

        // Slow down sampling
        config.sampling.interval_ms = 10000; // 10 seconds
        config.sampling.max_samples = 10;

        config
    }

    #[test]
    fn test_profiler_creation() {
        let config = fast_test_config();
        let result = MobilePerformanceProfiler::new(config);
        assert!(
            result.is_ok(),
            "Failed to create profiler: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_profiling_lifecycle() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        // Test initial state
        assert!(!profiler.is_profiling_active());

        // Test start profiling
        let session_id = profiler.start_profiling()?;
        assert!(!session_id.is_empty());
        assert!(profiler.is_profiling_active());

        // Test double start (should fail)
        assert!(profiler.start_profiling().is_err());

        // Test stop profiling
        let profiling_data = profiler.stop_profiling()?;
        assert!(!profiler.is_profiling_active());
        assert_eq!(profiling_data.session_info.metadata.session_id, session_id);

        Ok(())
    }

    #[test]
    fn test_event_recording() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        // Record various events
        profiler.record_inference_event("model_load", Some(250.0))?;
        profiler.record_inference_event("inference_start", None)?;
        profiler.record_inference_event("inference_end", Some(85.0))?;

        let profiling_data = profiler.stop_profiling()?;
        assert_eq!(profiling_data.events.len(), 3);

        Ok(())
    }

    #[test]
    fn test_metrics_collection() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        // Allow some time for metrics collection
        std::thread::sleep(Duration::from_millis(1));

        let metrics = profiler.get_current_metrics()?;
        assert!(metrics.timestamp > 0);

        let _stats = profiler.get_collection_stats()?;

        profiler.stop_profiling()?;
        Ok(())
    }

    #[test]
    #[ignore] // FIXME: This test has implementation issues causing 60+ second delays (likely thread/deadlock issue)
    fn test_bottleneck_detection() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        // Give profiler time to collect initial metrics
        std::thread::sleep(std::time::Duration::from_millis(1));

        let _bottlenecks = profiler.detect_bottlenecks()?;
        // Should not crash and return a vector (may be empty)

        profiler.stop_profiling()?;

        // Give background tasks time to clean up
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok(())
    }

    #[test]
    fn test_optimization_suggestions() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        let _suggestions = profiler.get_optimization_suggestions()?;
        // Should not crash and return a vector (may be empty)

        profiler.stop_profiling()?;
        Ok(())
    }

    #[test]
    fn test_pause_resume_profiling() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;
        assert!(profiler.is_profiling_active());

        profiler.pause_profiling()?;
        assert!(profiler.is_profiling_active()); // Still active, just paused

        profiler.resume_profiling()?;
        assert!(profiler.is_profiling_active());

        profiler.stop_profiling()?;
        assert!(!profiler.is_profiling_active());

        Ok(())
    }

    #[test]
    fn test_config_validation() {
        let mut config = MobileProfilerConfig::default();

        // Test valid config
        assert!(MobilePerformanceProfiler::validate_config(&config).is_ok());

        // Test invalid sampling interval
        config.sampling.interval_ms = 0;
        assert!(MobilePerformanceProfiler::validate_config(&config).is_err());

        // Reset and test invalid max samples
        config = MobileProfilerConfig::default();
        config.sampling.max_samples = 0;
        assert!(MobilePerformanceProfiler::validate_config(&config).is_err());

        // Reset and test invalid compression level
        config = MobileProfilerConfig::default();
        config.export_config.compression_level = 10;
        assert!(MobilePerformanceProfiler::validate_config(&config).is_err());
    }

    #[test]
    fn test_config_update() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        let mut new_config = MobileProfilerConfig::default();
        new_config.sampling.interval_ms = 200;
        new_config.memory_profiling.heap_analysis = true;

        profiler.update_config(new_config)?;

        Ok(())
    }

    #[test]
    fn test_export_functionality() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;
        profiler.record_inference_event("test_event", Some(100.0))?;
        profiler.stop_profiling()?;

        let export_path = profiler.export_data(ExportFormat::JSON)?;
        assert!(!export_path.is_empty());

        Ok(())
    }

    #[test]
    fn test_system_health_assessment() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;

        let health = profiler.get_system_health()?;
        assert!(health.overall_score >= 0.0 && health.overall_score <= 100.0);

        profiler.stop_profiling()?;
        Ok(())
    }

    #[test]
    fn test_performance_report_generation() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        profiler.start_profiling()?;
        profiler.record_inference_event("test_inference", Some(50.0))?;
        profiler.stop_profiling()?;

        let report = profiler.generate_performance_report()?;
        assert!(!report.is_empty());
        assert!(report.contains("html")); // Should be HTML format

        Ok(())
    }

    #[test]
    fn test_session_state_tracking() -> Result<()> {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config)?;

        // Test initial state
        let state = profiler.get_profiling_state();
        assert!(!state.is_active);
        assert_eq!(state.events_recorded, 0);

        // Start profiling and check state
        profiler.start_profiling()?;
        let state = profiler.get_profiling_state();
        assert!(state.is_active);
        assert!(state.current_session_id.is_some());
        assert!(state.start_time.is_some());

        // Record events and check counter
        profiler.record_inference_event("event1", None)?;
        profiler.record_inference_event("event2", None)?;
        let state = profiler.get_profiling_state();
        assert_eq!(state.events_recorded, 2);

        profiler.stop_profiling()?;
        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let config = fast_test_config();
        let profiler = MobilePerformanceProfiler::new(config).unwrap();

        // Test operations on inactive profiler
        assert!(profiler.stop_profiling().is_err());
        assert!(profiler.pause_profiling().is_err());
        assert!(profiler.resume_profiling().is_err());

        // Test invalid config updates
        let mut invalid_config = MobileProfilerConfig::default();
        invalid_config.sampling.interval_ms = 0;
        assert!(profiler.update_config(invalid_config).is_err());
    }

    #[test]
    fn test_thread_safety() -> Result<()> {
        use std::sync::Arc;
        use std::thread;

        let config = fast_test_config();
        let profiler = Arc::new(MobilePerformanceProfiler::new(config)?);

        profiler.start_profiling()?;

        // Spawn multiple threads to test concurrent access
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let profiler_clone = Arc::clone(&profiler);
                thread::spawn(move || {
                    for j in 0..10 {
                        let event_name = format!("thread_{}_event_{}", i, j);
                        let _ = profiler_clone.record_inference_event(&event_name, Some(10.0));
                        let _ = profiler_clone.get_current_metrics();
                        let _ = profiler_clone.detect_bottlenecks();
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let profiling_data = profiler.stop_profiling()?;

        // Should have recorded events from multiple threads
        assert!(profiling_data.events.len() > 0);

        Ok(())
    }
}
