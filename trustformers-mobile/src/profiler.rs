//! Advanced Mobile Performance Profiler
//!
//! This module provides comprehensive performance profiling capabilities for mobile ML
//! workloads, integrating with platform-specific tools and providing detailed performance
//! analysis, bottleneck detection, and optimization recommendations.

use crate::device_info::{MobileDeviceInfo, PerformanceTier};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Advanced mobile performance profiler
pub struct MobilePerformanceProfiler {
    config: ProfilerConfig,
    platform_profiler: Box<dyn PlatformProfiler + Send + Sync>,
    metrics_collector: MetricsCollector,
    bottleneck_detector: BottleneckDetector,
    performance_analyzer: PerformanceAnalyzer,
    alert_system: AlertSystem,
    profiling_session: Option<ProfilingSession>,
    historical_data: VecDeque<ProfileSnapshot>,
}

/// Performance profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable real-time profiling
    pub enable_realtime_profiling: bool,
    /// Profiling interval (ms)
    pub profiling_interval_ms: u64,
    /// Enable platform-specific profiler integration
    pub enable_platform_integration: bool,
    /// Maximum profile history size
    pub max_history_size: usize,
    /// Performance metrics to collect
    pub metrics_config: MetricsConfig,
    /// Bottleneck detection configuration
    pub bottleneck_config: BottleneckConfig,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Export profiling data
    pub enable_export: bool,
    /// Export format
    pub export_format: ExportFormat,
}

/// Platform-specific profiler trait
pub trait PlatformProfiler {
    /// Start platform-specific profiling
    fn start_profiling(&mut self) -> Result<()>;

    /// Stop platform-specific profiling
    fn stop_profiling(&mut self) -> Result<()>;

    /// Collect platform-specific metrics
    fn collect_metrics(&self) -> Result<PlatformMetrics>;

    /// Export profiling data
    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>>;

    /// Get platform capabilities
    fn get_capabilities(&self) -> Vec<ProfilerCapability>;
}

/// Platform-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlatformMetrics {
    /// CPU metrics
    pub cpu_metrics: CpuMetrics,
    /// GPU metrics
    pub gpu_metrics: Option<GpuMetrics>,
    /// Memory metrics
    pub memory_metrics: MemoryMetrics,
    /// Network metrics
    pub network_metrics: NetworkMetrics,
    /// Platform-specific metrics
    pub platform_specific: HashMap<String, f64>,
}

/// CPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// CPU utilization percentage
    pub utilization_percent: f32,
    /// Per-core utilization
    pub per_core_utilization: Vec<f32>,
    /// CPU frequency (MHz)
    pub frequency_mhz: Vec<u32>,
    /// Context switches per second
    pub context_switches_per_sec: u32,
    /// CPU load average
    pub load_average: [f32; 3],
    /// Time spent in user mode (%)
    pub user_time_percent: f32,
    /// Time spent in kernel mode (%)
    pub kernel_time_percent: f32,
    /// Time spent idle (%)
    pub idle_time_percent: f32,
}

/// GPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization_percent: f32,
    /// GPU memory utilization (%)
    pub memory_utilization_percent: f32,
    /// GPU frequency (MHz)
    pub frequency_mhz: u32,
    /// GPU temperature (°C)
    pub temperature_celsius: f32,
    /// GPU power consumption (mW)
    pub power_consumption_mw: f32,
    /// Number of active shaders
    pub active_shaders: u32,
}

/// Memory performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total memory usage (MB)
    pub total_usage_mb: usize,
    /// Available memory (MB)
    pub available_mb: usize,
    /// Memory pressure level
    pub pressure_level: MemoryPressureLevel,
    /// Page faults per second
    pub page_faults_per_sec: u32,
    /// Memory allocations per second
    pub allocations_per_sec: u32,
    /// Memory deallocations per second
    pub deallocations_per_sec: u32,
    /// Garbage collection metrics
    pub gc_metrics: Option<GcMetrics>,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Garbage collection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcMetrics {
    /// Total GC time (ms)
    pub total_gc_time_ms: u64,
    /// GC frequency (per minute)
    pub gc_frequency_per_min: f32,
    /// Average GC pause time (ms)
    pub avg_pause_time_ms: f32,
    /// Memory freed by GC (MB)
    pub memory_freed_mb: usize,
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Network bytes sent (per second)
    pub bytes_sent_per_sec: u64,
    /// Network bytes received (per second)
    pub bytes_received_per_sec: u64,
    /// Network latency (ms)
    pub latency_ms: f32,
    /// Connection count
    pub connection_count: u32,
    /// Network errors per second
    pub errors_per_sec: u32,
    /// Connection type
    pub connection_type: NetworkConnectionType,
    /// Signal strength (dBm)
    pub signal_strength_dbm: Option<i32>,
}

/// Network connection types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkConnectionType {
    WiFi,
    Cellular4G,
    Cellular5G,
    Ethernet,
    Offline,
    Unknown,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Collect CPU metrics
    pub collect_cpu: bool,
    /// Collect GPU metrics
    pub collect_gpu: bool,
    /// Collect memory metrics
    pub collect_memory: bool,
    /// Collect network metrics
    pub collect_network: bool,
    /// Collect inference-specific metrics
    pub collect_inference: bool,
    /// Sampling rate (Hz)
    pub sampling_rate_hz: u32,
    /// Collect detailed metrics
    pub detailed_collection: bool,
}

/// Metrics collector
struct MetricsCollector {
    config: MetricsConfig,
    device_info: MobileDeviceInfo,
    metrics_history: VecDeque<MetricsSnapshot>,
    collection_start_time: Option<Instant>,
}

/// Metrics snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub platform_metrics: PlatformMetrics,
    pub inference_metrics: InferenceMetrics,
    pub thermal_metrics: Option<ThermalMetrics>,
    pub battery_metrics: Option<BatteryPowerMetrics>,
}

/// Inference-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// Inference latency (ms)
    pub latency_ms: f32,
    /// Throughput (inferences per second)
    pub throughput_ips: f32,
    /// Queue depth
    pub queue_depth: usize,
    /// Model loading time (ms)
    pub model_load_time_ms: f32,
    /// Memory footprint (MB)
    pub memory_footprint_mb: usize,
    /// Accuracy score
    pub accuracy_score: Option<f32>,
    /// Backend utilization
    pub backend_utilization: BackendUtilization,
}

/// Backend utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendUtilization {
    /// CPU backend utilization (%)
    pub cpu_percent: f32,
    /// GPU backend utilization (%)
    pub gpu_percent: Option<f32>,
    /// NPU backend utilization (%)
    pub npu_percent: Option<f32>,
    /// Custom backend utilization (%)
    pub custom_percent: Option<f32>,
}

/// Thermal performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    /// Current temperature (°C)
    pub temperature_celsius: f32,
    /// Thermal throttling level
    pub throttling_level: u8,
    /// Thermal pressure
    pub thermal_pressure: f32,
    /// Cooling effectiveness
    pub cooling_effectiveness: f32,
}

/// Battery and power metrics for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryPowerMetrics {
    /// Current power consumption (mW)
    pub power_consumption_mw: f32,
    /// Battery drain rate (%/hour)
    pub drain_rate_percent_per_hour: f32,
    /// Power efficiency (inferences per mWh)
    pub power_efficiency: f32,
    /// Estimated battery life (minutes)
    pub estimated_life_minutes: u32,
}

/// Bottleneck detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckConfig {
    /// Enable CPU bottleneck detection
    pub detect_cpu_bottlenecks: bool,
    /// Enable memory bottleneck detection
    pub detect_memory_bottlenecks: bool,
    /// Enable I/O bottleneck detection
    pub detect_io_bottlenecks: bool,
    /// Enable thermal bottleneck detection
    pub detect_thermal_bottlenecks: bool,
    /// CPU threshold for bottleneck (%)
    pub cpu_threshold_percent: f32,
    /// Memory threshold for bottleneck (%)
    pub memory_threshold_percent: f32,
    /// Analysis window size
    pub analysis_window_samples: usize,
}

/// Bottleneck detector
struct BottleneckDetector {
    config: BottleneckConfig,
    analysis_buffer: VecDeque<MetricsSnapshot>,
    detected_bottlenecks: Vec<PerformanceBottleneck>,
}

/// Performance bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Description of the bottleneck
    pub description: String,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Duration of bottleneck (ms)
    pub duration_ms: u64,
    /// Impact on performance (%)
    pub performance_impact_percent: f32,
    /// Suggested optimizations
    pub optimizations: Vec<OptimizationSuggestion>,
    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    GPU,
    IO,
    Network,
    Thermal,
    Battery,
    Backend,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Optimization category
    pub category: OptimizationCategory,
    /// Description of the optimization
    pub description: String,
    /// Expected performance improvement (%)
    pub expected_improvement_percent: f32,
    /// Implementation difficulty
    pub difficulty: OptimizationDifficulty,
    /// Priority level
    pub priority: OptimizationPriority,
}

/// Optimization categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationCategory {
    ModelCompression,
    MemoryOptimization,
    ComputeOptimization,
    ThermalManagement,
    PowerOptimization,
    NetworkOptimization,
    CacheOptimization,
}

/// Optimization difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Optimization priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance analyzer for pattern detection and insights
struct PerformanceAnalyzer {
    performance_patterns: Vec<PerformancePattern>,
    regression_detector: RegressionDetector,
    trend_analyzer: TrendAnalyzer,
}

/// Performance pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Frequency of occurrence
    pub frequency: f32,
    /// Performance impact
    pub impact: f32,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Types of performance patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    MemoryLeak,
    CpuSpike,
    ThermalThrottling,
    BatteryDrain,
    NetworkCongestion,
    LoadBalanceIssue,
    CacheInefficiency,
}

/// Performance regression detector
struct RegressionDetector {
    baseline_metrics: Option<MetricsSnapshot>,
    regression_threshold_percent: f32,
    detected_regressions: Vec<PerformanceRegression>,
}

/// Performance regression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Metric that regressed
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f32,
    /// Current value
    pub current_value: f32,
    /// Regression percentage
    pub regression_percent: f32,
    /// Regression severity
    pub severity: RegressionSeverity,
}

/// Regression severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,
    Moderate,
    Major,
    Critical,
}

/// Performance trend analyzer
struct TrendAnalyzer {
    trend_window_size: usize,
    performance_trends: HashMap<String, PerformanceTrend>,
}

/// Performance trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend magnitude
    pub magnitude: f32,
    /// Trend confidence
    pub confidence: f32,
    /// Prediction for next period
    pub prediction: Option<f32>,
}

/// Trend directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Alert system for performance monitoring
struct AlertSystem {
    thresholds: AlertThresholds,
    active_alerts: Vec<PerformanceAlert>,
    alert_history: VecDeque<PerformanceAlert>,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU utilization alert threshold (%)
    pub cpu_threshold_percent: f32,
    /// Memory utilization alert threshold (%)
    pub memory_threshold_percent: f32,
    /// Latency alert threshold (ms)
    pub latency_threshold_ms: f32,
    /// Temperature alert threshold (°C)
    pub temperature_threshold_celsius: f32,
    /// Battery level alert threshold (%)
    pub battery_threshold_percent: u8,
    /// Power consumption alert threshold (mW)
    pub power_threshold_mw: f32,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Metric value that triggered alert
    pub trigger_value: f32,
    /// Threshold that was exceeded
    pub threshold_value: f32,
    /// Timestamp when alert was triggered
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Duration of the condition
    pub duration_ms: u64,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Alert types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    HighLatency,
    HighTemperature,
    LowBattery,
    HighPowerConsumption,
    PerformanceRegression,
    SystemOverload,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Profiling session information
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    session_id: String,
    start_time: Instant,
    end_time: Option<Instant>,
    config: ProfilerConfig,
    collected_snapshots: usize,
    session_stats: SessionStats,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    /// Total profiling duration (ms)
    pub duration_ms: u64,
    /// Total snapshots collected
    pub snapshots_collected: usize,
    /// Average sampling rate (Hz)
    pub avg_sampling_rate_hz: f32,
    /// Data size collected (bytes)
    pub data_size_bytes: usize,
    /// Bottlenecks detected
    pub bottlenecks_detected: usize,
    /// Alerts triggered
    pub alerts_triggered: usize,
}

/// Profile snapshot for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSnapshot {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub performance_score: f32,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub alerts: Vec<PerformanceAlert>,
    pub metrics: MetricsSnapshot,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// Profiler capabilities for different platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilerCapability {
    CpuProfiling,
    GpuProfiling,
    MemoryProfiling,
    NetworkProfiling,
    ThermalProfiling,
    BatteryProfiling,
    InstrumentsIntegration,
    SystraceIntegration,
    PerfettoIntegration,
    CustomProfiling,
}

/// Export formats for profiling data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    CSV,
    Protobuf,
    Trace,
    Chrome,
    Instruments,
    Perfetto,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_realtime_profiling: true,
            profiling_interval_ms: 1000, // 1 second
            enable_platform_integration: true,
            max_history_size: 1000,
            metrics_config: MetricsConfig::default(),
            bottleneck_config: BottleneckConfig::default(),
            alert_thresholds: AlertThresholds::default(),
            enable_export: true,
            export_format: ExportFormat::JSON,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collect_cpu: true,
            collect_gpu: true,
            collect_memory: true,
            collect_network: true,
            collect_inference: true,
            sampling_rate_hz: 10, // 10 Hz
            detailed_collection: false,
        }
    }
}

impl Default for BottleneckConfig {
    fn default() -> Self {
        Self {
            detect_cpu_bottlenecks: true,
            detect_memory_bottlenecks: true,
            detect_io_bottlenecks: true,
            detect_thermal_bottlenecks: true,
            cpu_threshold_percent: 80.0,
            memory_threshold_percent: 85.0,
            analysis_window_samples: 100,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold_percent: 90.0,
            memory_threshold_percent: 90.0,
            latency_threshold_ms: 500.0,
            temperature_threshold_celsius: 85.0,
            battery_threshold_percent: 20,
            power_threshold_mw: 5000.0, // 5W
        }
    }
}

impl MobilePerformanceProfiler {
    /// Create new performance profiler
    pub fn new(config: ProfilerConfig, device_info: &MobileDeviceInfo) -> Result<Self> {
        let platform_profiler = Self::create_platform_profiler(device_info)?;
        let metrics_collector =
            MetricsCollector::new(config.metrics_config.clone(), device_info.clone());
        let bottleneck_detector = BottleneckDetector::new(config.bottleneck_config.clone());
        let performance_analyzer = PerformanceAnalyzer::new();
        let alert_system = AlertSystem::new(config.alert_thresholds.clone());

        Ok(Self {
            config: config.clone(),
            platform_profiler,
            metrics_collector,
            bottleneck_detector,
            performance_analyzer,
            alert_system,
            profiling_session: None,
            historical_data: VecDeque::with_capacity(config.max_history_size),
        })
    }

    /// Create platform-specific profiler
    fn create_platform_profiler(
        device_info: &MobileDeviceInfo,
    ) -> Result<Box<dyn PlatformProfiler + Send + Sync>> {
        match device_info.basic_info.platform {
            crate::MobilePlatform::Ios => Ok(Box::new(IOSProfiler::new()?)),
            crate::MobilePlatform::Android => Ok(Box::new(AndroidProfiler::new()?)),
            crate::MobilePlatform::Generic => Ok(Box::new(GenericProfiler::new()?)),
        }
    }

    /// Start profiling session
    pub fn start_session(&mut self, session_id: String) -> Result<()> {
        if self.profiling_session.is_some() {
            return Err(TrustformersError::config_error(
                "Profiling session already active",
                "start_session",
            )
            .into());
        }

        self.profiling_session = Some(ProfilingSession {
            session_id: session_id.clone(),
            start_time: Instant::now(),
            end_time: None,
            config: self.config.clone(),
            collected_snapshots: 0,
            session_stats: SessionStats {
                duration_ms: 0,
                snapshots_collected: 0,
                avg_sampling_rate_hz: 0.0,
                data_size_bytes: 0,
                bottlenecks_detected: 0,
                alerts_triggered: 0,
            },
        });

        if self.config.enable_platform_integration {
            self.platform_profiler.start_profiling()?;
        }

        self.metrics_collector.start()?;

        Ok(())
    }

    /// Stop profiling session
    pub fn stop_session(&mut self) -> Result<SessionStats> {
        let session = self.profiling_session.take().ok_or_else(|| {
            TrustformersError::config_error("No active profiling session", "stop_session")
        })?;

        if self.config.enable_platform_integration {
            self.platform_profiler.stop_profiling()?;
        }

        self.metrics_collector.stop()?;

        let duration = session.start_time.elapsed();
        let stats = SessionStats {
            duration_ms: duration.as_millis() as u64,
            snapshots_collected: session.collected_snapshots,
            avg_sampling_rate_hz: session.collected_snapshots as f32 / duration.as_secs() as f32,
            data_size_bytes: self.estimate_data_size(),
            bottlenecks_detected: self.bottleneck_detector.detected_bottlenecks.len(),
            alerts_triggered: self.alert_system.alert_history.len(),
        };

        Ok(stats)
    }

    /// Collect performance snapshot
    pub fn collect_snapshot(&mut self) -> Result<ProfileSnapshot> {
        let platform_metrics = if self.config.enable_platform_integration {
            self.platform_profiler.collect_metrics()?
        } else {
            PlatformMetrics::default()
        };

        let metrics_snapshot = self.metrics_collector.collect_snapshot(platform_metrics)?;

        // Detect bottlenecks
        let bottlenecks = self.bottleneck_detector.analyze(&metrics_snapshot)?;

        // Check for alerts
        let alerts = self.alert_system.check_thresholds(&metrics_snapshot)?;

        // Generate optimization suggestions
        let optimizations = self
            .performance_analyzer
            .generate_suggestions(&metrics_snapshot, &bottlenecks)?;

        // Calculate performance score
        let performance_score = self.calculate_performance_score(&metrics_snapshot, &bottlenecks);

        let snapshot = ProfileSnapshot {
            timestamp: Instant::now(),
            performance_score,
            bottlenecks,
            alerts,
            metrics: metrics_snapshot,
            optimization_suggestions: optimizations,
        };

        // Update profiling session
        if let Some(ref mut session) = self.profiling_session {
            session.collected_snapshots += 1;
        }

        // Store in historical data
        self.historical_data.push_back(snapshot.clone());
        if self.historical_data.len() > self.config.max_history_size {
            self.historical_data.pop_front();
        }

        Ok(snapshot)
    }

    /// Calculate overall performance score (0.0-100.0)
    fn calculate_performance_score(
        &self,
        metrics: &MetricsSnapshot,
        bottlenecks: &[PerformanceBottleneck],
    ) -> f32 {
        let mut score = 100.0;

        // Penalize based on CPU utilization
        if metrics.platform_metrics.cpu_metrics.utilization_percent > 80.0 {
            score -= (metrics.platform_metrics.cpu_metrics.utilization_percent - 80.0) * 0.5;
        }

        // Penalize based on memory pressure
        match metrics.platform_metrics.memory_metrics.pressure_level {
            MemoryPressureLevel::Medium => score -= 10.0,
            MemoryPressureLevel::High => score -= 25.0,
            MemoryPressureLevel::Critical => score -= 50.0,
            _ => {},
        }

        // Penalize based on inference latency
        if metrics.inference_metrics.latency_ms > 100.0 {
            score -= (metrics.inference_metrics.latency_ms - 100.0) * 0.1;
        }

        // Penalize based on bottlenecks
        for bottleneck in bottlenecks {
            let penalty = match bottleneck.severity {
                BottleneckSeverity::Low => 5.0,
                BottleneckSeverity::Medium => 10.0,
                BottleneckSeverity::High => 20.0,
                BottleneckSeverity::Critical => 40.0,
            };
            score -= penalty;
        }

        score.max(0.0).min(100.0)
    }

    /// Estimate total data size collected
    fn estimate_data_size(&self) -> usize {
        // Rough estimate based on historical data size
        self.historical_data.len() * 2048 // ~2KB per snapshot
    }

    /// Export profiling data
    pub fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::JSON => {
                let data = serde_json::to_vec(&self.historical_data)
                    .map_err(|e| TrustformersError::serialization_error(e.to_string()))?;
                Ok(data)
            },
            _ => {
                // Delegate to platform profiler for specialized formats
                self.platform_profiler.export_data(format)
            },
        }
    }

    /// Get profiler capabilities
    pub fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.platform_profiler.get_capabilities()
    }

    /// Get current profiling statistics
    pub fn get_statistics(&self) -> Result<ProfilingStatistics> {
        let current_session = self.profiling_session.as_ref();

        Ok(ProfilingStatistics {
            total_snapshots: self.historical_data.len(),
            average_performance_score: self.calculate_average_performance_score(),
            active_bottlenecks: self.bottleneck_detector.detected_bottlenecks.len(),
            active_alerts: self.alert_system.active_alerts.len(),
            session_duration_ms: current_session
                .map(|s| s.start_time.elapsed().as_millis() as u64)
                .unwrap_or(0),
            data_collection_rate_hz: self.calculate_data_collection_rate(),
        })
    }

    fn calculate_average_performance_score(&self) -> f32 {
        if self.historical_data.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.historical_data.iter().map(|s| s.performance_score).sum();
        sum / self.historical_data.len() as f32
    }

    fn calculate_data_collection_rate(&self) -> f32 {
        if let Some(session) = &self.profiling_session {
            let duration_secs = session.start_time.elapsed().as_secs_f32();
            if duration_secs > 0.0 {
                return session.collected_snapshots as f32 / duration_secs;
            }
        }
        0.0
    }
}

/// Overall profiling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingStatistics {
    /// Total snapshots collected
    pub total_snapshots: usize,
    /// Average performance score
    pub average_performance_score: f32,
    /// Number of active bottlenecks
    pub active_bottlenecks: usize,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Current session duration (ms)
    pub session_duration_ms: u64,
    /// Data collection rate (Hz)
    pub data_collection_rate_hz: f32,
}

// Platform-specific profiler implementations
pub struct IOSProfiler {
    instruments_integration: bool,
    capabilities: Vec<ProfilerCapability>,
}

pub struct AndroidProfiler {
    systrace_integration: bool,
    perfetto_integration: bool,
    capabilities: Vec<ProfilerCapability>,
}

pub struct GenericProfiler {
    capabilities: Vec<ProfilerCapability>,
}

// Implement platform-specific profilers
impl IOSProfiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            instruments_integration: Self::check_instruments_availability(),
            capabilities: vec![
                ProfilerCapability::CpuProfiling,
                ProfilerCapability::MemoryProfiling,
                ProfilerCapability::GpuProfiling,
                ProfilerCapability::ThermalProfiling,
                ProfilerCapability::BatteryProfiling,
                ProfilerCapability::InstrumentsIntegration,
            ],
        })
    }

    fn check_instruments_availability() -> bool {
        // Check if Instruments tools are available
        #[cfg(target_os = "ios")]
        {
            // Platform-specific check for Instruments
            true // Placeholder
        }
        #[cfg(not(target_os = "ios"))]
        {
            false
        }
    }
}

impl PlatformProfiler for IOSProfiler {
    fn start_profiling(&mut self) -> Result<()> {
        // Start iOS-specific profiling using Core Animation Time Profiler, etc.
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<()> {
        // Stop iOS-specific profiling
        Ok(())
    }

    fn collect_metrics(&self) -> Result<PlatformMetrics> {
        // Collect iOS-specific metrics
        Ok(PlatformMetrics::default())
    }

    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::Instruments => {
                // Export in Instruments format
                Ok(vec![])
            },
            _ => Err(TrustformersError::config_error(
                "Export format not supported on iOS",
                "export_ios_data",
            )
            .into()),
        }
    }

    fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.capabilities.clone()
    }
}

impl AndroidProfiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            systrace_integration: Self::check_systrace_availability(),
            perfetto_integration: Self::check_perfetto_availability(),
            capabilities: vec![
                ProfilerCapability::CpuProfiling,
                ProfilerCapability::MemoryProfiling,
                ProfilerCapability::GpuProfiling,
                ProfilerCapability::NetworkProfiling,
                ProfilerCapability::SystraceIntegration,
                ProfilerCapability::PerfettoIntegration,
            ],
        })
    }

    fn check_systrace_availability() -> bool {
        // Check if systrace is available
        #[cfg(target_os = "android")]
        {
            true // Placeholder
        }
        #[cfg(not(target_os = "android"))]
        {
            false
        }
    }

    fn check_perfetto_availability() -> bool {
        // Check if Perfetto is available
        #[cfg(target_os = "android")]
        {
            true // Placeholder
        }
        #[cfg(not(target_os = "android"))]
        {
            false
        }
    }
}

impl PlatformProfiler for AndroidProfiler {
    fn start_profiling(&mut self) -> Result<()> {
        // Start Android-specific profiling using systrace, Perfetto, etc.
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<()> {
        // Stop Android-specific profiling
        Ok(())
    }

    fn collect_metrics(&self) -> Result<PlatformMetrics> {
        // Collect Android-specific metrics
        Ok(PlatformMetrics::default())
    }

    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::Trace | ExportFormat::Perfetto => {
                // Export in systrace/Perfetto format
                Ok(vec![])
            },
            _ => Err(TrustformersError::config_error(
                "Export format not supported on Android",
                "export_android_data",
            )
            .into()),
        }
    }

    fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.capabilities.clone()
    }
}

impl GenericProfiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            capabilities: vec![
                ProfilerCapability::CpuProfiling,
                ProfilerCapability::MemoryProfiling,
                ProfilerCapability::NetworkProfiling,
                ProfilerCapability::CustomProfiling,
            ],
        })
    }
}

impl PlatformProfiler for GenericProfiler {
    fn start_profiling(&mut self) -> Result<()> {
        // Start generic profiling
        Ok(())
    }

    fn stop_profiling(&mut self) -> Result<()> {
        // Stop generic profiling
        Ok(())
    }

    fn collect_metrics(&self) -> Result<PlatformMetrics> {
        // Collect generic metrics
        Ok(PlatformMetrics::default())
    }

    fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::JSON | ExportFormat::CSV => {
                // Export in generic formats
                Ok(vec![])
            },
            _ => Err(TrustformersError::config_error(
                "Export format not supported",
                "export_profiling_data",
            )
            .into()),
        }
    }

    fn get_capabilities(&self) -> Vec<ProfilerCapability> {
        self.capabilities.clone()
    }
}

impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            utilization_percent: 0.0,
            per_core_utilization: vec![],
            frequency_mhz: vec![],
            context_switches_per_sec: 0,
            load_average: [0.0, 0.0, 0.0],
            user_time_percent: 0.0,
            kernel_time_percent: 0.0,
            idle_time_percent: 100.0,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            total_usage_mb: 0,
            available_mb: 0,
            pressure_level: MemoryPressureLevel::Low,
            page_faults_per_sec: 0,
            allocations_per_sec: 0,
            deallocations_per_sec: 0,
            gc_metrics: None,
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            bytes_sent_per_sec: 0,
            bytes_received_per_sec: 0,
            latency_ms: 0.0,
            connection_count: 0,
            errors_per_sec: 0,
            connection_type: NetworkConnectionType::Unknown,
            signal_strength_dbm: None,
        }
    }
}

// Implementation stubs for internal components
impl MetricsCollector {
    fn new(config: MetricsConfig, device_info: MobileDeviceInfo) -> Self {
        Self {
            config,
            device_info,
            metrics_history: VecDeque::new(),
            collection_start_time: None,
        }
    }

    fn start(&mut self) -> Result<()> {
        self.collection_start_time = Some(Instant::now());
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.collection_start_time = None;
        Ok(())
    }

    fn collect_snapshot(&mut self, platform_metrics: PlatformMetrics) -> Result<MetricsSnapshot> {
        let snapshot = MetricsSnapshot {
            timestamp: Instant::now(),
            platform_metrics,
            inference_metrics: InferenceMetrics::default(),
            thermal_metrics: None,
            battery_metrics: None,
        };

        self.metrics_history.push_back(snapshot.clone());
        Ok(snapshot)
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            latency_ms: 0.0,
            throughput_ips: 0.0,
            queue_depth: 0,
            model_load_time_ms: 0.0,
            memory_footprint_mb: 0,
            accuracy_score: None,
            backend_utilization: BackendUtilization::default(),
        }
    }
}

impl Default for BackendUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            gpu_percent: None,
            npu_percent: None,
            custom_percent: None,
        }
    }
}

impl BottleneckDetector {
    fn new(config: BottleneckConfig) -> Self {
        Self {
            config,
            analysis_buffer: VecDeque::new(),
            detected_bottlenecks: vec![],
        }
    }

    fn analyze(&mut self, metrics: &MetricsSnapshot) -> Result<Vec<PerformanceBottleneck>> {
        self.analysis_buffer.push_back(metrics.clone());

        if self.analysis_buffer.len() > self.config.analysis_window_samples {
            self.analysis_buffer.pop_front();
        }

        // Analyze for bottlenecks
        let mut bottlenecks = vec![];

        // CPU bottleneck detection
        if self.config.detect_cpu_bottlenecks
            && metrics.platform_metrics.cpu_metrics.utilization_percent
                > self.config.cpu_threshold_percent
        {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CPU,
                description: format!(
                    "High CPU utilization detected: {:.1}%",
                    metrics.platform_metrics.cpu_metrics.utilization_percent
                ),
                severity: self.calculate_bottleneck_severity(
                    metrics.platform_metrics.cpu_metrics.utilization_percent,
                    self.config.cpu_threshold_percent,
                ),
                duration_ms: 0, // Would calculate based on analysis window
                performance_impact_percent: metrics
                    .platform_metrics
                    .cpu_metrics
                    .utilization_percent
                    - self.config.cpu_threshold_percent,
                optimizations: vec![OptimizationSuggestion {
                    category: OptimizationCategory::ComputeOptimization,
                    description: "Consider reducing inference frequency or using model compression"
                        .to_string(),
                    expected_improvement_percent: 20.0,
                    difficulty: OptimizationDifficulty::Medium,
                    priority: OptimizationPriority::High,
                }],
                confidence: 0.9,
            });
        }

        // Memory bottleneck detection
        if self.config.detect_memory_bottlenecks
            && matches!(
                metrics.platform_metrics.memory_metrics.pressure_level,
                MemoryPressureLevel::High | MemoryPressureLevel::Critical
            )
        {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                description: format!(
                    "High memory pressure detected: {:?}",
                    metrics.platform_metrics.memory_metrics.pressure_level
                ),
                severity: match metrics.platform_metrics.memory_metrics.pressure_level {
                    MemoryPressureLevel::High => BottleneckSeverity::High,
                    MemoryPressureLevel::Critical => BottleneckSeverity::Critical,
                    _ => BottleneckSeverity::Medium,
                },
                duration_ms: 0,
                performance_impact_percent: 30.0,
                optimizations: vec![OptimizationSuggestion {
                    category: OptimizationCategory::MemoryOptimization,
                    description: "Enable aggressive memory optimization and model quantization"
                        .to_string(),
                    expected_improvement_percent: 40.0,
                    difficulty: OptimizationDifficulty::Easy,
                    priority: OptimizationPriority::Critical,
                }],
                confidence: 0.95,
            });
        }

        self.detected_bottlenecks = bottlenecks.clone();
        Ok(bottlenecks)
    }

    fn calculate_bottleneck_severity(
        &self,
        current_value: f32,
        threshold: f32,
    ) -> BottleneckSeverity {
        let ratio = current_value / threshold;
        if ratio > 1.5 {
            BottleneckSeverity::Critical
        } else if ratio > 1.25 {
            BottleneckSeverity::High
        } else if ratio > 1.1 {
            BottleneckSeverity::Medium
        } else {
            BottleneckSeverity::Low
        }
    }
}

impl PerformanceAnalyzer {
    fn new() -> Self {
        Self {
            performance_patterns: vec![],
            regression_detector: RegressionDetector::new(),
            trend_analyzer: TrendAnalyzer::new(),
        }
    }

    fn generate_suggestions(
        &mut self,
        metrics: &MetricsSnapshot,
        bottlenecks: &[PerformanceBottleneck],
    ) -> Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = vec![];

        // Generate suggestions based on bottlenecks
        for bottleneck in bottlenecks {
            suggestions.extend(bottleneck.optimizations.clone());
        }

        // Generate suggestions based on metrics
        if metrics.inference_metrics.latency_ms > 200.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::ComputeOptimization,
                description: "High inference latency detected. Consider model optimization or hardware acceleration".to_string(),
                expected_improvement_percent: 50.0,
                difficulty: OptimizationDifficulty::Medium,
                priority: OptimizationPriority::High,
            });
        }

        Ok(suggestions)
    }
}

impl RegressionDetector {
    fn new() -> Self {
        Self {
            baseline_metrics: None,
            regression_threshold_percent: 10.0,
            detected_regressions: vec![],
        }
    }
}

impl TrendAnalyzer {
    fn new() -> Self {
        Self {
            trend_window_size: 50,
            performance_trends: HashMap::new(),
        }
    }
}

impl AlertSystem {
    fn new(thresholds: AlertThresholds) -> Self {
        Self {
            thresholds,
            active_alerts: vec![],
            alert_history: VecDeque::new(),
        }
    }

    fn check_thresholds(&mut self, metrics: &MetricsSnapshot) -> Result<Vec<PerformanceAlert>> {
        let mut alerts = vec![];

        // CPU threshold check
        if metrics.platform_metrics.cpu_metrics.utilization_percent
            > self.thresholds.cpu_threshold_percent
        {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::HighCpuUsage,
                severity: AlertSeverity::Warning,
                message: format!(
                    "CPU utilization is {:.1}%, exceeding threshold of {:.1}%",
                    metrics.platform_metrics.cpu_metrics.utilization_percent,
                    self.thresholds.cpu_threshold_percent
                ),
                trigger_value: metrics.platform_metrics.cpu_metrics.utilization_percent,
                threshold_value: self.thresholds.cpu_threshold_percent,
                timestamp: Instant::now(),
                duration_ms: 0,
                suggested_actions: vec![
                    "Reduce inference frequency".to_string(),
                    "Enable CPU throttling".to_string(),
                    "Optimize model computation".to_string(),
                ],
            });
        }

        // Memory threshold check
        let memory_usage_percent = if metrics.platform_metrics.memory_metrics.total_usage_mb > 0 {
            (metrics.platform_metrics.memory_metrics.total_usage_mb as f32
                / (metrics.platform_metrics.memory_metrics.total_usage_mb
                    + metrics.platform_metrics.memory_metrics.available_mb)
                    as f32)
                * 100.0
        } else {
            0.0
        };

        if memory_usage_percent > self.thresholds.memory_threshold_percent {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::HighMemoryUsage,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Memory utilization is {:.1}%, exceeding threshold of {:.1}%",
                    memory_usage_percent, self.thresholds.memory_threshold_percent
                ),
                trigger_value: memory_usage_percent,
                threshold_value: self.thresholds.memory_threshold_percent,
                timestamp: Instant::now(),
                duration_ms: 0,
                suggested_actions: vec![
                    "Enable memory optimization".to_string(),
                    "Reduce model size".to_string(),
                    "Clear model cache".to_string(),
                ],
            });
        }

        // Latency threshold check
        if metrics.inference_metrics.latency_ms > self.thresholds.latency_threshold_ms {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::HighLatency,
                severity: AlertSeverity::Error,
                message: format!(
                    "Inference latency is {:.1}ms, exceeding threshold of {:.1}ms",
                    metrics.inference_metrics.latency_ms, self.thresholds.latency_threshold_ms
                ),
                trigger_value: metrics.inference_metrics.latency_ms,
                threshold_value: self.thresholds.latency_threshold_ms,
                timestamp: Instant::now(),
                duration_ms: 0,
                suggested_actions: vec![
                    "Enable hardware acceleration".to_string(),
                    "Optimize model architecture".to_string(),
                    "Reduce batch size".to_string(),
                ],
            });
        }

        // Store alerts in history
        for alert in &alerts {
            self.alert_history.push_back(alert.clone());
        }

        self.active_alerts = alerts.clone();
        Ok(alerts)
    }
}

/// Utility functions for mobile performance profiling
pub struct MobileProfilerUtils;

impl MobileProfilerUtils {
    /// Create optimized profiler configuration for device
    pub fn create_optimized_config(device_info: &MobileDeviceInfo) -> ProfilerConfig {
        let mut config = ProfilerConfig::default();

        // Adjust based on device performance tier
        match device_info.performance_scores.overall_tier {
            PerformanceTier::VeryLow => {
                config.profiling_interval_ms = 10000; // 10 seconds
                config.metrics_config.sampling_rate_hz = 0; // Disabled
                config.max_history_size = 50;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Low => {
                config.profiling_interval_ms = 8000; // 8 seconds
                config.metrics_config.sampling_rate_hz = 0; // Disabled
                config.max_history_size = 75;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Budget => {
                config.profiling_interval_ms = 5000; // 5 seconds
                config.metrics_config.sampling_rate_hz = 1; // 1 Hz
                config.max_history_size = 100;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Medium => {
                config.profiling_interval_ms = 3000; // 3 seconds
                config.metrics_config.sampling_rate_hz = 2; // 2 Hz
                config.max_history_size = 300;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Mid => {
                config.profiling_interval_ms = 2000; // 2 seconds
                config.metrics_config.sampling_rate_hz = 5; // 5 Hz
                config.max_history_size = 500;
            },
            PerformanceTier::High => {
                config.profiling_interval_ms = 1000; // 1 second
                config.metrics_config.sampling_rate_hz = 10; // 10 Hz
                config.max_history_size = 1000;
                config.metrics_config.detailed_collection = true;
            },
            PerformanceTier::VeryHigh => {
                config.profiling_interval_ms = 750; // 750ms
                config.metrics_config.sampling_rate_hz = 15; // 15 Hz
                config.max_history_size = 1500;
                config.metrics_config.detailed_collection = true;
            },
            PerformanceTier::Flagship => {
                config.profiling_interval_ms = 500; // 500ms
                config.metrics_config.sampling_rate_hz = 20; // 20 Hz
                config.max_history_size = 2000;
                config.metrics_config.detailed_collection = true;
            },
        }

        // Enable GPU profiling if available
        if device_info.gpu_info.is_some() {
            config.metrics_config.collect_gpu = true;
        }

        config
    }

    /// Calculate performance efficiency score
    pub fn calculate_efficiency_score(metrics: &MetricsSnapshot) -> f32 {
        let cpu_efficiency = 100.0 - metrics.platform_metrics.cpu_metrics.utilization_percent;
        let memory_efficiency = match metrics.platform_metrics.memory_metrics.pressure_level {
            MemoryPressureLevel::Low => 100.0,
            MemoryPressureLevel::Medium => 75.0,
            MemoryPressureLevel::High => 50.0,
            MemoryPressureLevel::Critical => 25.0,
        };

        let inference_efficiency = if metrics.inference_metrics.latency_ms > 0.0 {
            (1000.0 / metrics.inference_metrics.latency_ms).min(100.0)
        } else {
            100.0
        };

        (cpu_efficiency + memory_efficiency + inference_efficiency) / 3.0
    }

    /// Export profiling data to Chrome trace format
    pub fn export_to_chrome_trace(snapshots: &[ProfileSnapshot]) -> Result<String> {
        // Implementation for Chrome trace format export
        let trace_data = json!({
            "traceEvents": snapshots.iter().map(|snapshot| {
                json!({
                    "name": "Performance Snapshot",
                    "ph": "X",
                    "ts": 0, // Would convert Instant to microseconds
                    "dur": 1000,
                    "pid": 1,
                    "tid": 1,
                    "args": {
                        "performance_score": snapshot.performance_score,
                        "cpu_usage": snapshot.metrics.platform_metrics.cpu_metrics.utilization_percent,
                        "memory_pressure": snapshot.metrics.platform_metrics.memory_metrics.pressure_level,
                        "inference_latency": snapshot.metrics.inference_metrics.latency_ms
                    }
                })
            }).collect::<Vec<_>>()
        });

        Ok(trace_data.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_info::{BasicDeviceInfo, CpuInfo, MemoryInfo, PerformanceScores};

    fn create_test_device_info() -> MobileDeviceInfo {
        MobileDeviceInfo {
            platform: crate::MobilePlatform::Generic,
            basic_info: BasicDeviceInfo {
                platform: crate::MobilePlatform::Generic,
                manufacturer: "Test".to_string(),
                model: "TestDevice".to_string(),
                os_version: "1.0".to_string(),
                hardware_id: "test123".to_string(),
                device_generation: Some(2023),
            },
            cpu_info: CpuInfo {
                architecture: "arm64".to_string(),
                total_cores: 8,
                core_count: 8,
                performance_cores: 4,
                efficiency_cores: 4,
                max_frequency_mhz: Some(3000),
                l1_cache_kb: Some(64),
                l2_cache_kb: Some(512),
                l3_cache_kb: Some(8192),
                features: vec!["NEON".to_string()],
                simd_support: crate::device_info::SimdSupport::Advanced,
            },
            memory_info: MemoryInfo {
                total_mb: 4096,
                available_mb: 2048,
                total_memory: 4096,
                available_memory: 2048,
                bandwidth_mbps: Some(25600),
                memory_type: "LPDDR5".to_string(),
                frequency_mhz: Some(6400),
                is_low_memory_device: false,
            },
            gpu_info: None,
            npu_info: None,
            thermal_info: crate::device_info::ThermalInfo {
                current_state: crate::device_info::ThermalState::Nominal,
                state: crate::device_info::ThermalState::Nominal,
                throttling_supported: true,
                temperature_sensors: vec![],
                thermal_zones: vec![],
            },
            power_info: crate::device_info::PowerInfo {
                battery_capacity_mah: Some(3000),
                battery_level_percent: Some(75),
                battery_level: Some(75),
                battery_health_percent: Some(95),
                charging_status: crate::device_info::ChargingStatus::NotCharging,
                is_charging: false,
                power_save_mode: false,
                low_power_mode_available: true,
            },
            available_backends: vec![crate::MobileBackend::CPU],
            performance_scores: PerformanceScores {
                cpu_single_core: Some(1200),
                cpu_multi_core: Some(8500),
                gpu_score: None,
                memory_score: Some(9200),
                overall_tier: PerformanceTier::High,
                tier: PerformanceTier::High,
            },
        }
    }

    #[test]
    fn test_profiler_creation() {
        let device_info = create_test_device_info();
        let config = ProfilerConfig::default();

        let profiler = MobilePerformanceProfiler::new(config, &device_info);
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_profiler_config_defaults() {
        let config = ProfilerConfig::default();
        assert!(config.enable_realtime_profiling);
        assert_eq!(config.profiling_interval_ms, 1000);
        assert!(config.enable_platform_integration);
        assert_eq!(config.max_history_size, 1000);
    }

    #[test]
    fn test_metrics_config_defaults() {
        let config = MetricsConfig::default();
        assert!(config.collect_cpu);
        assert!(config.collect_gpu);
        assert!(config.collect_memory);
        assert!(config.collect_network);
        assert!(config.collect_inference);
        assert_eq!(config.sampling_rate_hz, 10);
    }

    #[test]
    fn test_bottleneck_config_defaults() {
        let config = BottleneckConfig::default();
        assert!(config.detect_cpu_bottlenecks);
        assert!(config.detect_memory_bottlenecks);
        assert!(config.detect_io_bottlenecks);
        assert!(config.detect_thermal_bottlenecks);
        assert_eq!(config.cpu_threshold_percent, 80.0);
        assert_eq!(config.memory_threshold_percent, 85.0);
    }

    #[test]
    fn test_alert_thresholds_defaults() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.cpu_threshold_percent, 90.0);
        assert_eq!(thresholds.memory_threshold_percent, 90.0);
        assert_eq!(thresholds.latency_threshold_ms, 500.0);
        assert_eq!(thresholds.temperature_threshold_celsius, 85.0);
        assert_eq!(thresholds.battery_threshold_percent, 20);
        assert_eq!(thresholds.power_threshold_mw, 5000.0);
    }

    #[test]
    fn test_performance_score_calculation() {
        let device_info = create_test_device_info();
        let config = ProfilerConfig::default();
        let profiler =
            MobilePerformanceProfiler::new(config, &device_info).expect("Operation failed");

        let metrics = MetricsSnapshot {
            timestamp: Instant::now(),
            platform_metrics: PlatformMetrics::default(),
            inference_metrics: InferenceMetrics::default(),
            thermal_metrics: None,
            battery_metrics: None,
        };

        let bottlenecks = vec![];
        let score = profiler.calculate_performance_score(&metrics, &bottlenecks);
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_optimized_config_generation() {
        let device_info = create_test_device_info();
        let config = MobileProfilerUtils::create_optimized_config(&device_info);

        // Should be optimized for high-performance device
        assert_eq!(config.profiling_interval_ms, 1000);
        assert_eq!(config.metrics_config.sampling_rate_hz, 10);
        assert_eq!(config.max_history_size, 1000);
        assert!(config.metrics_config.detailed_collection);
    }

    #[test]
    fn test_efficiency_score_calculation() {
        let metrics = MetricsSnapshot {
            timestamp: Instant::now(),
            platform_metrics: PlatformMetrics {
                cpu_metrics: CpuMetrics {
                    utilization_percent: 50.0,
                    ..Default::default()
                },
                memory_metrics: MemoryMetrics {
                    pressure_level: MemoryPressureLevel::Low,
                    ..Default::default()
                },
                ..Default::default()
            },
            inference_metrics: InferenceMetrics {
                latency_ms: 100.0,
                ..Default::default()
            },
            thermal_metrics: None,
            battery_metrics: None,
        };

        let score = MobileProfilerUtils::calculate_efficiency_score(&metrics);
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_platform_profiler_capabilities() {
        let ios_profiler = IOSProfiler::new().expect("Operation failed");
        let capabilities = ios_profiler.get_capabilities();
        assert!(capabilities.contains(&ProfilerCapability::CpuProfiling));
        assert!(capabilities.contains(&ProfilerCapability::InstrumentsIntegration));

        let android_profiler = AndroidProfiler::new().expect("Operation failed");
        let capabilities = android_profiler.get_capabilities();
        assert!(capabilities.contains(&ProfilerCapability::CpuProfiling));
        assert!(capabilities.contains(&ProfilerCapability::SystraceIntegration));
    }

    #[test]
    fn test_bottleneck_severity_calculation() {
        let config = BottleneckConfig::default();
        let detector = BottleneckDetector::new(config);

        let severity = detector.calculate_bottleneck_severity(120.0, 80.0);
        assert_eq!(severity, BottleneckSeverity::High);

        let severity = detector.calculate_bottleneck_severity(160.0, 80.0);
        assert_eq!(severity, BottleneckSeverity::Critical);
    }

    #[test]
    fn test_memory_pressure_levels() {
        assert!(MemoryPressureLevel::Critical > MemoryPressureLevel::High);
        assert!(MemoryPressureLevel::High > MemoryPressureLevel::Medium);
        assert!(MemoryPressureLevel::Medium > MemoryPressureLevel::Low);
    }

    #[test]
    fn test_export_format_serialization() {
        let format = ExportFormat::JSON;
        let serialized = serde_json::to_string(&format).expect("Operation failed");
        let deserialized: ExportFormat =
            serde_json::from_str(&serialized).expect("Operation failed");
        assert_eq!(format, deserialized);
    }
}
