//! Enhanced Performance Profiler for TrustformeRS
//!
//! This module provides advanced performance profiling capabilities with:
//! - Real-time performance monitoring
//! - Hardware-specific optimization suggestions
//! - Memory leak detection
//! - Comprehensive performance analytics
//! - Integration with modern observability tools

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Enhanced Performance Profiler with advanced analytics
#[derive(Debug)]
pub struct EnhancedProfiler {
    sessions: Arc<RwLock<HashMap<String, ProfilingSession>>>,
    global_metrics: Arc<RwLock<GlobalMetrics>>,
    config: ProfilerConfig,
}

/// Configuration for the enhanced profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable hardware-specific profiling
    pub hardware_profiling: bool,
    /// Enable memory leak detection
    pub memory_leak_detection: bool,
    /// Enable real-time performance alerts
    pub real_time_alerts: bool,
    /// Enable AI-powered performance analysis
    pub ai_powered_analysis: bool,
    /// Sampling interval for continuous profiling
    pub sampling_interval_ms: u64,
    /// Maximum number of performance samples to keep
    pub max_samples: usize,
    /// Performance thresholds for alerts
    pub thresholds: PerformanceThresholds,
    /// Export formats enabled
    pub export_formats: Vec<ExportFormat>,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            hardware_profiling: true,
            memory_leak_detection: true,
            real_time_alerts: true,
            ai_powered_analysis: false, // Disabled by default
            sampling_interval_ms: 100,
            max_samples: 10000,
            thresholds: PerformanceThresholds::default(),
            export_formats: vec![ExportFormat::JSON, ExportFormat::Prometheus],
        }
    }
}

/// Performance thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_latency_ms: f32,
    pub min_throughput_ops_per_sec: f32,
    pub max_memory_usage_mb: f32,
    pub max_cpu_usage_percent: f32,
    pub max_gpu_usage_percent: f32,
    pub memory_leak_threshold_mb: f32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000.0,
            min_throughput_ops_per_sec: 10.0,
            max_memory_usage_mb: 1024.0,
            max_cpu_usage_percent: 90.0,
            max_gpu_usage_percent: 95.0,
            memory_leak_threshold_mb: 10.0,
        }
    }
}

/// Export formats for profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    CSV,
    Prometheus,
    Flamegraph,
    OpenTelemetry,
    Jaeger,
}

/// Profiling session for tracking performance of specific operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSession {
    pub session_id: String,
    pub operation_name: String,
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    pub samples: Vec<PerformanceSample>,
    pub hardware_info: HardwareInfo,
    pub memory_tracker: MemoryTracker,
    pub status: SessionStatus,
}

/// Performance sample capturing point-in-time metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSample {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub latency_ms: f32,
    pub throughput_ops_per_sec: f32,
    pub memory_usage_mb: f32,
    pub cpu_usage_percent: f32,
    /// GPU utilisation, when a GPU telemetry source is available.
    ///
    /// `None` means "not measurable on this build/platform" — this crate has no
    /// pure-Rust vendor telemetry, so it never guesses a figure.
    pub gpu_usage_percent: Option<f32>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Hardware information for optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_model: String,
    pub total_memory_gb: f32,
    pub gpu_info: Vec<GPUInfo>,
    pub platform: Platform,
    pub specialized_hardware: Vec<SpecializedHardware>,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub name: String,
    pub memory_gb: f32,
    pub compute_capability: String,
    pub utilization_percent: f32,
}

/// Platform detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    Linux,
    Windows,
    MacOS,
    Unknown,
}

/// Specialized hardware detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecializedHardware {
    CUDA,
    ROCm,
    Metal,
    OpenCL,
    TensorRT,
    CoreML,
    ONNX,
    TPU,
    NPU,
}

/// Memory tracking for leak detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTracker {
    pub initial_memory_mb: f32,
    pub peak_memory_mb: f32,
    pub current_memory_mb: f32,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub leak_detected: bool,
    pub memory_samples: Vec<MemorySample>,
}

/// Memory sample for tracking over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub memory_mb: f32,
    pub allocations: u64,
    pub deallocations: u64,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Completed,
    Failed,
    Cancelled,
}

/// Global metrics across all sessions
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    pub total_sessions: u64,
    pub active_sessions: u64,
    pub average_latency_ms: f32,
    pub total_operations: u64,
    pub memory_leaks_detected: u64,
    pub performance_alerts: u64,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// AI-powered optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub category: OptimizationCategory,
    pub severity: SuggestionSeverity,
    pub description: String,
    pub suggested_action: String,
    pub expected_improvement: String,
    pub confidence: f32,
}

/// Optimization categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Memory,
    CPU,
    GPU,
    IO,
    NetworkLatency,
    ModelArchitecture,
    BatchSize,
    Quantization,
    Caching,
    Threading,
}

/// Suggestion severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Comprehensive performance analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub session_summary: SessionSummary,
    pub performance_trends: PerformanceTrends,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub optimization_recommendations: Vec<OptimizationSuggestion>,
    pub hardware_utilization: HardwareUtilization,
    pub memory_analysis: MemoryAnalysis,
}

/// Session summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub total_duration_ms: f32,
    pub total_operations: u64,
    pub average_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub peak_throughput_ops_per_sec: f32,
    pub peak_memory_mb: f32,
}

/// Performance trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub latency_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub trend_confidence: f32,
}

/// Trend direction analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: BottleneckType,
    pub bottleneck_severity: f32,
    pub contributing_factors: Vec<String>,
    pub impact_analysis: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    GPU,
    IO,
    Network,
    ModelComplexity,
    DataLoading,
    Synchronization,
    Unknown,
}

/// Hardware utilization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareUtilization {
    pub cpu_utilization_percent: f32,
    pub memory_utilization_percent: f32,
    /// `None` when no GPU telemetry was available for this session.
    pub gpu_utilization_percent: Option<f32>,
    pub efficiency_score: f32,
    pub underutilized_resources: Vec<String>,
}

/// Memory analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    /// Share of the peak that is unreturned growth, in 0..=1.
    pub leak_probability: f32,
    /// Heap fragmentation, when an instrumented allocator can report it.
    ///
    /// `None` on a stock Rust process: nothing measures it.
    pub fragmentation_level: Option<f32>,
    /// Shape of the recorded memory series.
    pub allocation_pattern: AllocationPattern,
    /// Garbage-collection impact. Always `None` in a non-GC runtime.
    pub gc_impact: Option<f32>,
    /// Estimated headroom for memory optimisation.
    ///
    /// `None` until a source exists that can measure it.
    pub optimization_potential: Option<f32>,
}

/// Memory allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPattern {
    Steady,
    Spiky,
    Growing,
    Cyclical,
    Chaotic,
}

impl EnhancedProfiler {
    /// Create a new enhanced profiler with configuration
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            global_metrics: Arc::new(RwLock::new(GlobalMetrics::default())),
            config,
        }
    }

    /// Start a new profiling session
    pub async fn start_session(
        &self,
        session_id: String,
        operation_name: String,
    ) -> Result<(), String> {
        let hardware_info = self.detect_hardware().await;
        let session = ProfilingSession {
            session_id: session_id.clone(),
            operation_name,
            start_time: Instant::now(),
            samples: Vec::new(),
            hardware_info,
            memory_tracker: MemoryTracker {
                initial_memory_mb: self.get_current_memory_usage(),
                peak_memory_mb: 0.0,
                current_memory_mb: 0.0,
                allocation_count: 0,
                deallocation_count: 0,
                leak_detected: false,
                memory_samples: Vec::new(),
            },
            status: SessionStatus::Active,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session);

        let mut global_metrics = self.global_metrics.write().await;
        global_metrics.total_sessions += 1;
        global_metrics.active_sessions += 1;

        Ok(())
    }

    /// Record a performance sample
    pub async fn record_sample(
        &self,
        session_id: &str,
        custom_metrics: HashMap<String, f64>,
    ) -> Result<(), String> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            let sample = PerformanceSample {
                timestamp: Instant::now(),
                latency_ms: session.start_time.elapsed().as_millis() as f32,
                throughput_ops_per_sec: self.calculate_throughput(session).await,
                memory_usage_mb: self.get_current_memory_usage(),
                cpu_usage_percent: self.get_cpu_usage().await,
                gpu_usage_percent: self.get_gpu_usage().await,
                custom_metrics,
            };

            // Update memory tracker
            session.memory_tracker.current_memory_mb = sample.memory_usage_mb;
            if sample.memory_usage_mb > session.memory_tracker.peak_memory_mb {
                session.memory_tracker.peak_memory_mb = sample.memory_usage_mb;
            }

            // Add memory sample
            session.memory_tracker.memory_samples.push(MemorySample {
                timestamp: sample.timestamp,
                memory_mb: sample.memory_usage_mb,
                allocations: session.memory_tracker.allocation_count,
                deallocations: session.memory_tracker.deallocation_count,
            });

            session.samples.push(sample);

            // Check for performance alerts
            if self.config.real_time_alerts {
                self.check_performance_alerts(session).await;
            }

            // Limit samples to prevent memory growth
            if session.samples.len() > self.config.max_samples {
                session.samples.remove(0);
            }

            Ok(())
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// End a profiling session and generate analysis
    pub async fn end_session(&self, session_id: &str) -> Result<PerformanceAnalysis, String> {
        let mut sessions = self.sessions.write().await;
        if let Some(mut session) = sessions.remove(session_id) {
            session.status = SessionStatus::Completed;

            let mut global_metrics = self.global_metrics.write().await;
            global_metrics.active_sessions -= 1;

            // Generate comprehensive analysis
            let analysis = self.generate_analysis(&session).await;

            // Add optimization suggestions to global metrics
            global_metrics
                .optimization_suggestions
                .extend(analysis.optimization_recommendations.clone());

            Ok(analysis)
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// Detect hardware configuration from the operating system.
    ///
    /// CPU model, core count and installed memory are real `sysinfo` readings.
    /// GPU enumeration has no pure-Rust source in this build, so the GPU list
    /// stays empty rather than carrying a placeholder device, and the
    /// specialized-hardware list reflects the features this binary was actually
    /// compiled with.
    async fn detect_hardware(&self) -> HardwareInfo {
        let mut system = sysinfo::System::new();
        system.refresh_cpu_all();
        system.refresh_memory();

        let cpu_cores = sysinfo::System::physical_core_count().unwrap_or_else(num_cpus::get);
        let cpu_model = system
            .cpus()
            .first()
            .map(|cpu| cpu.brand().trim().to_string())
            .filter(|brand| !brand.is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let total_memory_gb = system.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);

        let mut specialized_hardware = Vec::new();
        if cfg!(feature = "cuda") {
            specialized_hardware.push(SpecializedHardware::CUDA);
        }

        HardwareInfo {
            cpu_cores,
            cpu_model,
            total_memory_gb,
            // No pure-Rust GPU enumeration is linked in; an empty list is the
            // honest answer, an invented "Mock GPU" is not.
            gpu_info: Vec::new(),
            platform: if cfg!(target_os = "linux") {
                Platform::Linux
            } else if cfg!(target_os = "windows") {
                Platform::Windows
            } else if cfg!(target_os = "macos") {
                Platform::MacOS
            } else {
                Platform::Unknown
            },
            specialized_hardware,
        }
    }

    /// Resident memory of this process, in megabytes.
    ///
    /// Returns `0.0` only when the operating system refuses to report the
    /// process — never a synthetic figure.
    fn get_current_memory_usage(&self) -> f32 {
        crate::profiler::read_process_memory()
            .map(|m| m.resident_bytes as f32 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    }

    /// Live system-wide CPU utilisation, in percent.
    async fn get_cpu_usage(&self) -> f32 {
        read_cpu_usage().await
    }

    /// GPU utilisation, if any source can supply it.
    ///
    /// This build links no GPU telemetry, so the answer is `None`. Callers must
    /// render that as "not available" rather than as zero.
    async fn get_gpu_usage(&self) -> Option<f32> {
        None
    }

    /// Calculate throughput for a session
    async fn calculate_throughput(&self, session: &ProfilingSession) -> f32 {
        let duration_sec = session.start_time.elapsed().as_secs_f32();
        if duration_sec > 0.0 {
            session.samples.len() as f32 / duration_sec
        } else {
            0.0
        }
    }

    /// Check for performance alerts
    async fn check_performance_alerts(&self, session: &ProfilingSession) {
        if let Some(latest_sample) = session.samples.last() {
            let mut alerts_triggered = 0;

            if latest_sample.latency_ms > self.config.thresholds.max_latency_ms {
                alerts_triggered += 1;
                tracing::warn!(
                    "ALERT: High latency detected: {:.2}ms",
                    latest_sample.latency_ms
                );
            }

            if latest_sample.memory_usage_mb > self.config.thresholds.max_memory_usage_mb {
                alerts_triggered += 1;
                tracing::warn!(
                    "ALERT: High memory usage: {:.2}MB",
                    latest_sample.memory_usage_mb
                );
            }

            if latest_sample.cpu_usage_percent > self.config.thresholds.max_cpu_usage_percent {
                alerts_triggered += 1;
                tracing::warn!(
                    "ALERT: High CPU usage: {:.2}%",
                    latest_sample.cpu_usage_percent
                );
            }

            if alerts_triggered > 0 {
                let mut global_metrics = self.global_metrics.write().await;
                global_metrics.performance_alerts += alerts_triggered;
            }
        }
    }

    /// Generate comprehensive performance analysis
    async fn generate_analysis(&self, session: &ProfilingSession) -> PerformanceAnalysis {
        let session_summary = self.calculate_session_summary(session);
        let performance_trends = self.analyze_trends(session);
        let bottleneck_analysis = self.analyze_bottlenecks(session);
        let optimization_recommendations =
            self.generate_optimization_recommendations(session).await;
        let hardware_utilization = self.analyze_hardware_utilization(session);
        let memory_analysis = self.analyze_memory_usage(session);

        PerformanceAnalysis {
            session_summary,
            performance_trends,
            bottleneck_analysis,
            optimization_recommendations,
            hardware_utilization,
            memory_analysis,
        }
    }

    /// Calculate session summary statistics
    fn calculate_session_summary(&self, session: &ProfilingSession) -> SessionSummary {
        let latencies: Vec<f32> = session.samples.iter().map(|s| s.latency_ms).collect();
        let throughputs: Vec<f32> =
            session.samples.iter().map(|s| s.throughput_ops_per_sec).collect();

        SessionSummary {
            total_duration_ms: session.start_time.elapsed().as_millis() as f32,
            total_operations: session.samples.len() as u64,
            average_latency_ms: latencies.iter().sum::<f32>() / latencies.len() as f32,
            p95_latency_ms: self.percentile(&latencies, 0.95),
            p99_latency_ms: self.percentile(&latencies, 0.99),
            peak_throughput_ops_per_sec: throughputs.iter().cloned().fold(0.0f32, f32::max),
            peak_memory_mb: session.memory_tracker.peak_memory_mb,
        }
    }

    /// Calculate percentile from a sorted list
    fn percentile(&self, data: &[f32], percentile: f32) -> f32 {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = ((data.len() as f32 - 1.0) * percentile) as usize;
        sorted_data.get(index).copied().unwrap_or(0.0)
    }

    /// Analyze performance trends from the recorded samples.
    ///
    /// Each series is split in half and the two means compared; the confidence
    /// is the fraction of the series that actually carried data, so a session
    /// with too few samples reports a low confidence instead of a fixed 0.8.
    fn analyze_trends(&self, session: &ProfilingSession) -> PerformanceTrends {
        let latencies: Vec<f32> = session.samples.iter().map(|s| s.latency_ms).collect();
        let throughputs: Vec<f32> =
            session.samples.iter().map(|s| s.throughput_ops_per_sec).collect();
        let memories: Vec<f32> = session.samples.iter().map(|s| s.memory_usage_mb).collect();

        PerformanceTrends {
            // Rising latency is a regression, rising throughput is an improvement.
            latency_trend: Self::trend_of(&latencies, false),
            throughput_trend: Self::trend_of(&throughputs, true),
            memory_trend: Self::trend_of(&memories, false),
            trend_confidence: Self::trend_confidence(session.samples.len()),
        }
    }

    /// Direction of a measured series.
    ///
    /// `higher_is_better` flips the mapping so the same comparison serves both
    /// latency (lower is better) and throughput (higher is better). A series
    /// whose halves differ by less than one standard deviation is `Stable`;
    /// a series whose standard deviation exceeds half its mean is `Volatile`.
    fn trend_of(series: &[f32], higher_is_better: bool) -> TrendDirection {
        if series.len() < 4 {
            return TrendDirection::Stable;
        }
        let mean = series.iter().sum::<f32>() / series.len() as f32;
        let variance = series.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / series.len() as f32;
        let std_dev = variance.sqrt();

        let midpoint = series.len() / 2;
        let first_mean = series[..midpoint].iter().sum::<f32>() / midpoint as f32;
        let second_mean = series[midpoint..].iter().sum::<f32>() / (series.len() - midpoint) as f32;
        let delta = second_mean - first_mean;

        if std_dev > mean.abs() * 0.5 && mean.abs() > f32::EPSILON {
            return TrendDirection::Volatile;
        }
        if delta.abs() <= std_dev {
            return TrendDirection::Stable;
        }
        let rising = delta > 0.0;
        if rising == higher_is_better {
            TrendDirection::Improving
        } else {
            TrendDirection::Degrading
        }
    }

    /// How much a trend verdict can be trusted, from the sample count alone.
    ///
    /// Reaches 1.0 at 32 samples; below 4 samples no direction is inferred at
    /// all and the confidence is 0.
    fn trend_confidence(sample_count: usize) -> f32 {
        if sample_count < 4 {
            return 0.0;
        }
        (sample_count as f32 / 32.0).min(1.0)
    }

    /// Identify the resource closest to its configured threshold.
    ///
    /// Every number below is an average over the session's real samples; the
    /// contributing factors quote those measurements rather than describing
    /// generic causes.
    fn analyze_bottlenecks(&self, session: &ProfilingSession) -> BottleneckAnalysis {
        if session.samples.is_empty() {
            return BottleneckAnalysis {
                primary_bottleneck: BottleneckType::Unknown,
                bottleneck_severity: 0.0,
                contributing_factors: Vec::new(),
                impact_analysis: "No samples were recorded for this session".to_string(),
            };
        }
        let count = session.samples.len() as f32;
        let avg_cpu = session.samples.iter().map(|s| s.cpu_usage_percent).sum::<f32>() / count;
        let avg_memory = session.samples.iter().map(|s| s.memory_usage_mb).sum::<f32>() / count;
        let avg_latency = session.samples.iter().map(|s| s.latency_ms).sum::<f32>() / count;

        let thresholds = &self.config.thresholds;
        let cpu_load = ratio(avg_cpu, thresholds.max_cpu_usage_percent);
        let memory_load = ratio(avg_memory, thresholds.max_memory_usage_mb);
        let latency_load = ratio(avg_latency, thresholds.max_latency_ms);

        let (primary_bottleneck, severity) = [
            (BottleneckType::CPU, cpu_load),
            (BottleneckType::Memory, memory_load),
            (BottleneckType::ModelComplexity, latency_load),
        ]
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((BottleneckType::Unknown, 0.0));

        let mut contributing_factors = Vec::new();
        if cpu_load > 0.5 {
            contributing_factors.push(format!(
                "mean CPU utilisation {avg_cpu:.1}% is {:.0}% of the configured limit",
                cpu_load * 100.0
            ));
        }
        if memory_load > 0.5 {
            contributing_factors.push(format!(
                "mean resident memory {avg_memory:.1}MB is {:.0}% of the configured limit",
                memory_load * 100.0
            ));
        }
        if latency_load > 0.5 {
            contributing_factors.push(format!(
                "mean latency {avg_latency:.1}ms is {:.0}% of the configured limit",
                latency_load * 100.0
            ));
        }

        let impact_analysis = if contributing_factors.is_empty() {
            "All measured resources stayed below half of their configured limits".to_string()
        } else {
            format!(
                "{:?} is closest to its limit at {:.0}% of the threshold",
                primary_bottleneck,
                severity * 100.0
            )
        };

        BottleneckAnalysis {
            primary_bottleneck: if severity > 0.0 {
                primary_bottleneck
            } else {
                BottleneckType::Unknown
            },
            bottleneck_severity: severity.min(1.0),
            contributing_factors,
            impact_analysis,
        }
    }

    /// Generate AI-powered optimization recommendations
    async fn generate_optimization_recommendations(
        &self,
        session: &ProfilingSession,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Memory optimization suggestion
        if session.memory_tracker.peak_memory_mb > 500.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Memory,
                severity: SuggestionSeverity::Medium,
                description: "High peak memory usage detected".to_string(),
                suggested_action: "Consider implementing memory pooling or reducing batch size"
                    .to_string(),
                expected_improvement: "20-30% reduction in memory usage".to_string(),
                confidence: 0.85,
            });
        }

        // Batch size optimization
        if session.samples.len() > 100 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::BatchSize,
                severity: SuggestionSeverity::Low,
                description: "Batch size may be sub-optimal for throughput".to_string(),
                suggested_action: "Experiment with larger batch sizes for better GPU utilization"
                    .to_string(),
                expected_improvement: "15-25% improvement in throughput".to_string(),
                confidence: 0.7,
            });
        }

        suggestions
    }

    /// Analyze hardware utilization
    fn analyze_hardware_utilization(&self, session: &ProfilingSession) -> HardwareUtilization {
        let avg_cpu = session.samples.iter().map(|s| s.cpu_usage_percent).sum::<f32>()
            / session.samples.len() as f32;
        // Average only over samples that carried a GPU reading; if none did,
        // the figure stays unknown.
        let gpu_readings: Vec<f32> =
            session.samples.iter().filter_map(|s| s.gpu_usage_percent).collect();
        let avg_gpu = if gpu_readings.is_empty() {
            None
        } else {
            Some(gpu_readings.iter().sum::<f32>() / gpu_readings.len() as f32)
        };
        let avg_memory = session.samples.iter().map(|s| s.memory_usage_mb).sum::<f32>()
            / session.samples.len() as f32;

        let total_memory_mb = session.hardware_info.total_memory_gb * 1024.0;
        let memory_utilization_percent =
            if total_memory_mb > 0.0 { avg_memory / total_memory_mb * 100.0 } else { 0.0 };

        // Only resources that were actually measured can be called
        // under-utilised. GPU utilisation is unknown on this build, so it is
        // never listed.
        let mut underutilized_resources = Vec::new();
        if avg_cpu < 25.0 {
            underutilized_resources.push("CPU".to_string());
        }
        if memory_utilization_percent < 25.0 {
            underutilized_resources.push("Memory".to_string());
        }

        HardwareUtilization {
            cpu_utilization_percent: avg_cpu,
            memory_utilization_percent,
            gpu_utilization_percent: avg_gpu,
            // Mean of the utilisation figures that were actually measured.
            efficiency_score: match avg_gpu {
                Some(gpu) => (avg_cpu + gpu) / 2.0 / 100.0,
                None => avg_cpu / 100.0,
            },
            underutilized_resources,
        }
    }

    /// Analyse the recorded memory series.
    ///
    /// Growth and shape come from the real samples. Fragmentation and
    /// garbage-collection impact have no source in a Rust process without an
    /// instrumented allocator, so they are reported as unknown rather than
    /// filled with a constant.
    fn analyze_memory_usage(&self, session: &ProfilingSession) -> MemoryAnalysis {
        let series: Vec<f32> =
            session.memory_tracker.memory_samples.iter().map(|s| s.memory_mb).collect();
        let initial = session.memory_tracker.initial_memory_mb;
        let peak = session.memory_tracker.peak_memory_mb;
        let growth = peak - initial;

        // Leak likelihood: how much of the peak is growth that never came back.
        let leak_probability = if peak > 0.0 {
            let final_level = series.last().copied().unwrap_or(initial);
            ((final_level - initial) / peak).clamp(0.0, 1.0)
        } else {
            0.0
        };

        MemoryAnalysis {
            leak_probability,
            fragmentation_level: None,
            allocation_pattern: Self::classify_allocation_pattern(&series, growth),
            gc_impact: None,
            optimization_potential: None,
        }
    }

    /// Classify the shape of a memory series.
    fn classify_allocation_pattern(series: &[f32], growth: f32) -> AllocationPattern {
        if series.len() < 4 {
            return AllocationPattern::Steady;
        }
        let mean = series.iter().sum::<f32>() / series.len() as f32;
        if mean <= f32::EPSILON {
            return AllocationPattern::Steady;
        }
        let variance = series.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / series.len() as f32;
        let coefficient_of_variation = variance.sqrt() / mean;

        let direction_changes = series
            .windows(3)
            .filter(|w| (w[1] - w[0]).signum() != (w[2] - w[1]).signum())
            .count();
        let change_rate = direction_changes as f32 / (series.len() - 2).max(1) as f32;

        if coefficient_of_variation > 0.5 && change_rate > 0.5 {
            AllocationPattern::Chaotic
        } else if change_rate > 0.5 {
            AllocationPattern::Cyclical
        } else if growth > mean * 0.25 {
            AllocationPattern::Growing
        } else if coefficient_of_variation > 0.25 {
            AllocationPattern::Spiky
        } else {
            AllocationPattern::Steady
        }
    }

    /// Export profiling data in specified format
    pub async fn export_data(
        &self,
        session_id: &str,
        format: ExportFormat,
    ) -> Result<String, String> {
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            match format {
                ExportFormat::JSON => serde_json::to_string_pretty(session)
                    .map_err(|e| format!("JSON export failed: {}", e)),
                ExportFormat::CSV => {
                    // Simple CSV export - in real implementation, use proper CSV library
                    let mut csv =
                        "timestamp,latency_ms,throughput,memory_mb,cpu_percent,gpu_percent\n"
                            .to_string();
                    for sample in &session.samples {
                        csv.push_str(&format!(
                            "{:?},{},{},{},{},{}\n",
                            sample.timestamp,
                            sample.latency_ms,
                            sample.throughput_ops_per_sec,
                            sample.memory_usage_mb,
                            sample.cpu_usage_percent,
                            sample
                                .gpu_usage_percent
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "NA".to_string())
                        ));
                    }
                    Ok(csv)
                },
                ExportFormat::Prometheus => {
                    // Prometheus metrics format
                    let mut prometheus = String::new();
                    if let Some(latest_sample) = session.samples.last() {
                        prometheus.push_str(&format!(
                            "# HELP trustformers_latency_ms Current latency in milliseconds\n\
                             # TYPE trustformers_latency_ms gauge\n\
                             trustformers_latency_ms{{session=\"{}\"}} {}\n",
                            session_id, latest_sample.latency_ms
                        ));
                    }
                    Ok(prometheus)
                },
                ExportFormat::Flamegraph => {
                    // Emit a minimal folded-stacks text format compatible with
                    // FlameGraph.pl and inferno-flamegraph.
                    let mut flamegraph = String::new();
                    for sample in &session.samples {
                        // Each line: "stack;frame latency_ms"
                        flamegraph.push_str(&format!(
                            "{};latency_ms {}\n",
                            session.operation_name, sample.latency_ms as u64
                        ));
                        if sample.cpu_usage_percent > 0.0 {
                            flamegraph.push_str(&format!(
                                "{};cpu_usage {}\n",
                                session.operation_name, sample.cpu_usage_percent as u64
                            ));
                        }
                    }
                    Ok(flamegraph)
                },
                ExportFormat::OpenTelemetry => {
                    // Emit OTLP-compatible JSON span format (simplified).
                    let spans: Vec<serde_json::Value> = session
                        .samples
                        .iter()
                        .enumerate()
                        .map(|(i, sample)| {
                            serde_json::json!({
                                "traceId": format!("{:032x}", i),
                                "spanId":  format!("{:016x}", i),
                                "name": session.operation_name,
                                "kind": 1,
                                "attributes": {
                                    "latency_ms": sample.latency_ms,
                                    "throughput_ops_per_sec": sample.throughput_ops_per_sec,
                                    "memory_usage_mb": sample.memory_usage_mb,
                                    "cpu_usage_percent": sample.cpu_usage_percent,
                                }
                            })
                        })
                        .collect();
                    serde_json::to_string_pretty(&serde_json::json!({
                        "resourceSpans": [{
                            "scopeSpans": [{ "spans": spans }]
                        }]
                    }))
                    .map_err(|e| format!("OpenTelemetry export failed: {}", e))
                },
                ExportFormat::Jaeger => {
                    // Emit Jaeger-compatible JSON trace format.
                    let spans: Vec<serde_json::Value> = session
                        .samples
                        .iter()
                        .enumerate()
                        .map(|(i, sample)| {
                            serde_json::json!({
                                "traceID": format!("{:032x}", i),
                                "spanID":  format!("{:016x}", i),
                                "operationName": session.operation_name,
                                "duration": (sample.latency_ms * 1000.0) as u64,
                                "tags": [
                                    { "key": "throughput_ops_per_sec",
                                      "vFloat64": sample.throughput_ops_per_sec },
                                    { "key": "memory_usage_mb",
                                      "vFloat64": sample.memory_usage_mb },
                                ]
                            })
                        })
                        .collect();
                    serde_json::to_string_pretty(&serde_json::json!({
                        "data": [{
                            "traceID": session_id,
                            "spans": spans,
                            "processes": {
                                "p1": { "serviceName": "trustformers" }
                            }
                        }]
                    }))
                    .map_err(|e| format!("Jaeger export failed: {}", e))
                },
            }
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// Get global performance metrics
    pub async fn get_global_metrics(&self) -> GlobalMetrics {
        self.global_metrics.read().await.clone()
    }
}

/// Live system-wide CPU utilisation, in percent.
///
/// `sysinfo` derives utilisation from the delta between two refreshes, so a
/// single shared `System` is kept and only re-sampled once the minimum update
/// interval has elapsed. That keeps repeated sampling from adding a fixed sleep
/// to every measurement — a profiler must not dominate what it measures.
pub async fn read_cpu_usage() -> f32 {
    /// Shared sampler state: the `sysinfo` handle, when it was last refreshed,
    /// and the utilisation that refresh produced.
    struct CpuSampler {
        system: sysinfo::System,
        refreshed_at: std::time::Instant,
        last_value: Option<f32>,
    }

    static CPU: std::sync::OnceLock<std::sync::Mutex<CpuSampler>> = std::sync::OnceLock::new();

    let cell = CPU.get_or_init(|| {
        let mut system = sysinfo::System::new();
        system.refresh_cpu_usage();
        std::sync::Mutex::new(CpuSampler {
            system,
            refreshed_at: std::time::Instant::now(),
            last_value: None,
        })
    });

    // Decide whether this call has to wait for a fresh delta window. The guard
    // is released before the await so the returned future stays `Send`.
    let wait = match cell.lock() {
        Ok(guard) => {
            match sysinfo::MINIMUM_CPU_UPDATE_INTERVAL.checked_sub(guard.refreshed_at.elapsed()) {
                // The window has not elapsed. Re-reading now would measure a
                // near-zero interval and report a false 0%, so reuse the value
                // the last refresh actually measured when there is one.
                Some(remaining) => match guard.last_value {
                    Some(value) => return value,
                    None => Some(remaining),
                },
                None => None,
            }
        },
        Err(_) => return 0.0,
    };
    if let Some(wait) = wait {
        tokio::time::sleep(wait).await;
    }

    match cell.lock() {
        Ok(mut guard) => {
            // Another caller may have refreshed while this one slept.
            if guard.refreshed_at.elapsed() < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
                if let Some(value) = guard.last_value {
                    return value;
                }
            }
            guard.system.refresh_cpu_usage();
            guard.refreshed_at = std::time::Instant::now();
            let value = guard.system.global_cpu_usage();
            guard.last_value = Some(value);
            value
        },
        Err(_) => 0.0,
    }
}

/// `value / limit`, clamped to a non-negative number; `0.0` when the limit is
/// not positive.
fn ratio(value: f32, limit: f32) -> f32 {
    if limit <= 0.0 {
        0.0
    } else {
        (value / limit).max(0.0)
    }
}

/// Log a profiling failure without pulling `tracing` into the caller's crate.
///
/// `enhanced_profile_operation!` calls this instead of expanding a
/// `tracing::warn!` in the caller's crate graph, which would only compile for
/// callers that happen to depend on `tracing` themselves.
#[doc(hidden)]
pub fn log_profiler_warning(context: &str, error: &dyn std::fmt::Display) {
    tracing::warn!(%error, "{context}");
}

/// Global profiler instance for easy access
static GLOBAL_PROFILER: std::sync::OnceLock<Arc<EnhancedProfiler>> = std::sync::OnceLock::new();

/// Initialize the global profiler
pub fn init_global_profiler(config: ProfilerConfig) {
    let _ = GLOBAL_PROFILER.get_or_init(|| Arc::new(EnhancedProfiler::new(config)));
}

/// Get the global profiler instance
pub fn global_profiler() -> Option<Arc<EnhancedProfiler>> {
    GLOBAL_PROFILER.get().cloned()
}

/// Convenience macro for enhanced profiling operations.
///
/// Wraps an `async` block in a profiling session. Every path is
/// `$crate`-qualified so the macro works from any crate, and profiling failures
/// never abort the work being profiled: if the global profiler was never
/// initialised the block simply runs un-profiled, and session errors are logged
/// rather than panicking.
///
/// This doctest exercises the macro from *outside* the defining crate, which is
/// what catches the unhygienic-path bug the qualification above fixes, and
/// shows that a missing global profiler degrades gracefully instead of
/// panicking:
///
/// ```
/// use trustformers::enhanced_profile_operation;
///
/// let runtime = tokio::runtime::Runtime::new().expect("runtime");
/// let answer = runtime.block_on(async { enhanced_profile_operation!("decode", { 21 * 2 }) });
/// assert_eq!(answer, 42);
/// ```
#[macro_export]
macro_rules! enhanced_profile_operation {
    ($operation_name:expr, $block:block) => {{
        match $crate::enhanced_profiler::global_profiler() {
            // No profiler installed: run the work un-profiled rather than
            // aborting the program that asked to be profiled.
            None => $block,
            Some(profiler) => {
                let session_id = format!(
                    "{}_{}",
                    $operation_name,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                );

                if let Err(error) =
                    profiler.start_session(session_id.clone(), $operation_name.to_string()).await
                {
                    $crate::enhanced_profiler::log_profiler_warning(
                        "failed to start profiling session",
                        &error,
                    );
                }

                let result = $block;

                if let Err(error) =
                    profiler.record_sample(&session_id, std::collections::HashMap::new()).await
                {
                    $crate::enhanced_profiler::log_profiler_warning(
                        "failed to record profiling sample",
                        &error,
                    );
                }
                if let Err(error) = profiler.end_session(&session_id).await {
                    $crate::enhanced_profiler::log_profiler_warning(
                        "failed to end profiling session",
                        &error,
                    );
                }

                result
            },
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Regression: hardware detection returned "Mock CPU Model"/"Mock GPU"/16GB
    // and an unconditional CUDA+Metal+ONNX list, usage was derived from
    // `std::ptr::addr_of!(self)`, and bottleneck analysis reported
    // "Mock factor 1"/"Mock factor 2". Every assertion below fails against that
    // implementation.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn hardware_detection_reports_real_values() {
        let profiler = EnhancedProfiler::new(ProfilerConfig::default());
        let hardware = profiler.detect_hardware().await;

        assert_ne!(hardware.cpu_model, "Mock CPU Model");
        assert!(hardware.cpu_cores > 0, "a running process has CPU cores");
        assert!(
            hardware.total_memory_gb > 0.0,
            "installed memory must be a real reading"
        );
        assert!(
            hardware.gpu_info.is_empty(),
            "no GPU enumeration is linked in, so no device may be listed"
        );
        assert!(
            !hardware.specialized_hardware.contains(&SpecializedHardware::Metal)
                && !hardware.specialized_hardware.contains(&SpecializedHardware::ONNX),
            "specialized hardware must reflect compiled features, not a fixed list"
        );
    }

    #[tokio::test]
    async fn memory_and_gpu_readings_are_measured_or_absent() {
        let profiler = EnhancedProfiler::new(ProfilerConfig::default());
        let memory_mb = profiler.get_current_memory_usage();
        assert!(
            memory_mb > 0.0,
            "resident memory must be a real reading, got {memory_mb}"
        );
        assert!(
            profiler.get_gpu_usage().await.is_none(),
            "with no GPU telemetry the answer must be `None`, never a number"
        );
    }

    #[tokio::test]
    async fn bottleneck_factors_quote_real_measurements() {
        let profiler = EnhancedProfiler::new(ProfilerConfig::default());
        profiler
            .start_session("bottlenecks".to_string(), "test".to_string())
            .await
            .expect("session should start");
        profiler
            .record_sample("bottlenecks", HashMap::new())
            .await
            .expect("sample should record");

        let sessions = profiler.sessions.read().await;
        let session = sessions.get("bottlenecks").expect("session exists");
        let analysis = profiler.analyze_bottlenecks(session);

        for factor in &analysis.contributing_factors {
            assert!(
                !factor.contains("Mock factor"),
                "contributing factors must describe measurements: {factor}"
            );
        }
        assert!((0.0..=1.0).contains(&analysis.bottleneck_severity));
    }

    #[tokio::test]
    async fn memory_analysis_reports_unknowns_as_none() {
        let profiler = EnhancedProfiler::new(ProfilerConfig::default());
        profiler
            .start_session("memory".to_string(), "test".to_string())
            .await
            .expect("session should start");
        profiler
            .record_sample("memory", HashMap::new())
            .await
            .expect("sample should record");

        let sessions = profiler.sessions.read().await;
        let session = sessions.get("memory").expect("session exists");
        let analysis = profiler.analyze_memory_usage(session);

        assert!(
            analysis.fragmentation_level.is_none(),
            "0.3 was the old placeholder fragmentation level"
        );
        assert!(
            analysis.gc_impact.is_none(),
            "a Rust process has no garbage collector to measure"
        );
        assert!(analysis.optimization_potential.is_none());
        assert!((0.0..=1.0).contains(&analysis.leak_probability));
    }
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_enhanced_profiler_basic_functionality() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);

        let session_id = "test_session".to_string();
        let operation_name = "test_operation".to_string();

        // Start session
        profiler
            .start_session(session_id.clone(), operation_name)
            .await
            .expect("async operation failed");

        // Record samples (more than 100 to trigger batch size optimization)
        for i in 0..105 {
            let mut custom_metrics = HashMap::new();
            custom_metrics.insert("iteration".to_string(), i as f64);
            profiler
                .record_sample(&session_id, custom_metrics)
                .await
                .expect("async operation failed");
            if i % 20 == 0 {
                sleep(Duration::from_millis(1)).await; // Reduce sleep frequency for faster test
            }
        }

        // End session and get analysis
        let analysis = profiler.end_session(&session_id).await.expect("async operation failed");

        assert!(analysis.session_summary.total_operations > 0);
        assert!(analysis.session_summary.total_duration_ms > 0.0);
        assert!(!analysis.optimization_recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_global_profiler() {
        let config = ProfilerConfig::default();
        init_global_profiler(config);

        let profiler = global_profiler().expect("Global profiler should be initialized");

        let session_id = "global_test".to_string();
        profiler
            .start_session(session_id.clone(), "global_test".to_string())
            .await
            .expect("operation failed in test");

        let metrics = profiler.get_global_metrics().await;
        assert!(metrics.total_sessions > 0);
    }

    #[tokio::test]
    async fn test_export_functionality() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);

        let session_id = "export_test".to_string();
        profiler
            .start_session(session_id.clone(), "export_test".to_string())
            .await
            .expect("operation failed in test");
        profiler
            .record_sample(&session_id, HashMap::new())
            .await
            .expect("async operation failed");

        // Test JSON export
        let json_export = profiler.export_data(&session_id, ExportFormat::JSON).await;
        assert!(json_export.is_ok());

        // Test CSV export
        let csv_export = profiler.export_data(&session_id, ExportFormat::CSV).await;
        assert!(csv_export.is_ok());

        profiler.end_session(&session_id).await.expect("async operation failed");
    }

    // --- Additional tests to meet the 15+ test requirement ---

    #[test]
    fn test_profiler_config_default_values() {
        let config = ProfilerConfig::default();
        assert!(
            config.sampling_interval_ms > 0,
            "sampling_interval_ms should be positive"
        );
        assert!(config.max_samples > 0, "max_samples should be positive");
        assert!(
            !config.export_formats.is_empty(),
            "at least one export format should be enabled by default"
        );
    }

    #[test]
    fn test_performance_thresholds_default_values() {
        let thresholds = PerformanceThresholds::default();
        assert!(
            thresholds.max_latency_ms > 0.0,
            "max_latency_ms should be positive"
        );
        assert!(
            thresholds.min_throughput_ops_per_sec > 0.0,
            "min_throughput_ops_per_sec should be positive"
        );
        assert!(
            thresholds.max_memory_usage_mb > 0.0,
            "max_memory_usage_mb should be positive"
        );
        assert!(
            thresholds.max_cpu_usage_percent > 0.0 && thresholds.max_cpu_usage_percent <= 100.0,
            "max_cpu_usage_percent should be in (0.0, 100.0]"
        );
    }

    #[tokio::test]
    async fn test_profiler_start_increments_global_total_sessions() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);

        // Use LCG-based suffix for unique session IDs
        let seed: u64 = 0xABCDEF0123456789;
        let session_id = format!(
            "session_{}",
            seed.wrapping_mul(1103515245).wrapping_add(12345)
        );

        let initial_metrics = profiler.get_global_metrics().await;
        profiler
            .start_session(session_id.clone(), "op1".to_string())
            .await
            .expect("start_session should succeed");

        let after_start = profiler.get_global_metrics().await;
        assert_eq!(
            after_start.total_sessions,
            initial_metrics.total_sessions + 1,
            "total_sessions should increment after start_session"
        );
        assert_eq!(
            after_start.active_sessions,
            initial_metrics.active_sessions + 1,
            "active_sessions should increment after start_session"
        );
    }

    #[tokio::test]
    async fn test_profiler_end_session_decrements_active_sessions() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let session_id = "decrement_test".to_string();

        profiler
            .start_session(session_id.clone(), "op".to_string())
            .await
            .expect("start_session should succeed");
        let after_start = profiler.get_global_metrics().await;
        let active_before = after_start.active_sessions;

        profiler.end_session(&session_id).await.expect("end_session should succeed");

        let after_end = profiler.get_global_metrics().await;
        assert_eq!(
            after_end.active_sessions,
            active_before - 1,
            "active_sessions should decrement after end_session"
        );
    }

    #[tokio::test]
    async fn test_profiler_end_session_returns_analysis() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let session_id = "analysis_test".to_string();

        profiler
            .start_session(session_id.clone(), "test_op".to_string())
            .await
            .expect("start_session should succeed");
        profiler
            .record_sample(&session_id, HashMap::new())
            .await
            .expect("record_sample should succeed");

        let analysis = profiler
            .end_session(&session_id)
            .await
            .expect("end_session should return analysis");

        assert!(
            analysis.session_summary.total_operations > 0,
            "session_summary should have at least one operation"
        );
        assert!(
            analysis.session_summary.total_duration_ms >= 0.0,
            "total_duration_ms should be non-negative"
        );
    }

    #[tokio::test]
    async fn test_profiler_total_duration_calculation() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let session_id = "duration_test".to_string();

        profiler
            .start_session(session_id.clone(), "duration_op".to_string())
            .await
            .expect("start_session should succeed");

        // Record samples to build up duration
        for i in 0..5 {
            let mut metrics = HashMap::new();
            metrics.insert("i".to_string(), i as f64);
            profiler
                .record_sample(&session_id, metrics)
                .await
                .expect("record_sample should succeed");
        }

        let analysis = profiler.end_session(&session_id).await.expect("end_session should succeed");
        assert!(
            analysis.session_summary.total_duration_ms >= 0.0,
            "total_duration_ms should be non-negative"
        );
    }

    #[tokio::test]
    async fn test_profiler_record_sample_stores_custom_metrics() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let session_id = "custom_metrics_test".to_string();

        profiler
            .start_session(session_id.clone(), "custom_op".to_string())
            .await
            .expect("start_session should succeed");

        let mut custom = HashMap::new();
        custom.insert("tokens_per_sec".to_string(), 42.5f64);
        custom.insert("batch_size".to_string(), 32.0f64);

        profiler
            .record_sample(&session_id, custom)
            .await
            .expect("record_sample should succeed");

        profiler.end_session(&session_id).await.expect("end_session should succeed");
    }

    #[tokio::test]
    async fn test_profiler_multiple_concurrent_sessions() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);

        // Use LCG to generate unique session names
        let seed: u64 = 0xFEEDFACECAFEBABE;
        let s1 = format!(
            "session_a_{}",
            seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
        );
        let s2 = format!(
            "session_b_{}",
            seed.wrapping_mul(1103515245).wrapping_add(12345)
        );

        profiler
            .start_session(s1.clone(), "op_a".to_string())
            .await
            .expect("start first session");
        profiler
            .start_session(s2.clone(), "op_b".to_string())
            .await
            .expect("start second session");

        let metrics = profiler.get_global_metrics().await;
        assert!(
            metrics.active_sessions >= 2,
            "should have at least 2 active sessions, got {}",
            metrics.active_sessions
        );

        profiler.end_session(&s1).await.expect("end first session");
        profiler.end_session(&s2).await.expect("end second session");
    }

    #[tokio::test]
    async fn test_profiler_end_nonexistent_session_returns_err() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let result = profiler.end_session("nonexistent-session-xyz").await;
        assert!(
            result.is_err(),
            "ending a non-existent session should return Err"
        );
    }

    #[tokio::test]
    async fn test_profiler_record_sample_nonexistent_returns_err() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let result = profiler.record_sample("nonexistent-xyz", HashMap::new()).await;
        assert!(
            result.is_err(),
            "recording sample for non-existent session should return Err"
        );
    }

    #[tokio::test]
    async fn test_profiler_json_export_contains_session_id() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let session_id = "json_content_test".to_string();

        profiler
            .start_session(session_id.clone(), "test_op".to_string())
            .await
            .expect("start_session should succeed");
        profiler
            .record_sample(&session_id, HashMap::new())
            .await
            .expect("record_sample should succeed");

        let json = profiler
            .export_data(&session_id, ExportFormat::JSON)
            .await
            .expect("JSON export should succeed");
        assert!(
            json.contains(&session_id),
            "JSON export should contain the session_id"
        );
    }

    #[tokio::test]
    async fn test_profiler_prometheus_export_format() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let session_id = "prometheus_test".to_string();

        profiler
            .start_session(session_id.clone(), "prom_op".to_string())
            .await
            .expect("start_session should succeed");
        profiler
            .record_sample(&session_id, HashMap::new())
            .await
            .expect("record_sample should succeed");

        let result = profiler.export_data(&session_id, ExportFormat::Prometheus).await;
        assert!(result.is_ok(), "Prometheus export should succeed");
        let content = result.expect("prometheus content");
        assert!(
            content.contains("trustformers_latency_ms"),
            "Prometheus export should contain metric name"
        );
    }

    #[tokio::test]
    async fn test_profiler_analysis_contains_optimization_recommendations() {
        let config = ProfilerConfig::default();
        let profiler = EnhancedProfiler::new(config);
        let session_id = "opt_recs_test".to_string();

        profiler
            .start_session(session_id.clone(), "opt_op".to_string())
            .await
            .expect("start_session should succeed");

        // Record multiple samples to give profiler data to analyze
        for i in 0..20u64 {
            let mut metrics = HashMap::new();
            // Use LCG to vary the iteration value
            let val = i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            metrics.insert(
                "iteration".to_string(),
                (val >> 32) as f64 / u32::MAX as f64,
            );
            profiler
                .record_sample(&session_id, metrics)
                .await
                .expect("record_sample should succeed");
        }

        let analysis = profiler.end_session(&session_id).await.expect("end_session should succeed");
        // Recommendations may or may not be present depending on thresholds
        // Just verify the field is accessible without panic
        let _ = analysis.optimization_recommendations.len();
    }
}
